####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_read_user_dict_success_with_prompts. Retrieved 13/19 statements.
# Partially parsed test_read_user_dict_uses_var_name_as_question. Retrieved 7/13 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'user_data'
    var_4 = 'Enter details'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'Input: '
    var_8 = module_0.read_user_dict(var_3, var_6, var_5, var_7)
    var_9 = bool(var_8 == var_2)
    assert var_9 is True
    var_10 = 'your_module.JsonPrompt.ask'
    var_11 = 'Input: Enter details [cyan bold](default)[/]'
    var_12 = {}
    var_13 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = {}
    var_2 = module_0.read_user_dict(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = 'your_module.JsonPrompt.ask'
    var_5 = 'config [cyan bold](none)[/]'
    var_6 = {}
    var_7 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'not a dict'
    var_2 = module_0.read_user_dict(var_0, var_1)
    var_3 = 'TypeError not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_true_is_dir. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_no_input_true_is_file. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_with_input_delete_yes. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_with_input_delete_no_reuse_no_exit. Retrieved 3/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/file.zip'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)



# Parsed testcases at query #3
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._prompts_from_options(var_0)
    var_2 = bool(var_1 == {'__prompt__': 'Select a template'})
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = 'choice2'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = 'Option One'
    var_5 = 'The first option'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Option Two'
    var_8 = 'The second option'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = '__prompt__'
    var_12 = 'Select a template'
    var_13 = 'Option One (The first option)'
    var_14 = 'Option Two (The second option)'
    var_15 = {var_11: var_12, var_0: var_13, var_1: var_14}
    var_16 = module_0._prompts_from_options(var_10)
    var_17 = bool(var_16 == var_15)
    assert var_17 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = 'title'
    var_2 = 'description'
    var_3 = 'Same'
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = '__prompt__'
    var_7 = 'Select a template'
    var_8 = {var_6: var_7, var_0: var_3}
    var_9 = module_0._prompts_from_options(var_5)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = 'title'
    var_2 = 'Only Title'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '__prompt__'
    var_6 = 'Select a template'
    var_7 = 'Only Title (Only Title)'
    var_8 = {var_5: var_6, var_0: var_7}
    var_9 = module_0._prompts_from_options(var_4)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = 'description'
    var_2 = 'Only Description'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '__prompt__'
    var_6 = 'Select a template'
    var_7 = 'choice1 (Only Description)'
    var_8 = {var_5: var_6, var_0: var_7}
    var_9 = module_0._prompts_from_options(var_4)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '__prompt__'
    var_4 = 'Select a template'
    var_5 = {var_3: var_4, var_0: var_0}
    var_6 = module_0._prompts_from_options(var_2)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'choice1'
    var_1 = 'title'
    var_2 = 'description'
    var_3 = 123
    var_4 = 'Numeric title'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = '__prompt__'
    var_8 = 'Select a template'
    var_9 = '123 (Numeric title)'
    var_10 = {var_7: var_8, var_0: var_9}
    var_11 = module_0._prompts_from_options(var_6)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True



# Parsed testcases at query #4
#--------------------------




import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = None
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 is None

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = True
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 is True
    var_4 = False
    var_5 = {}
    var_6 = module_1.render_variable(var_0, var_4, var_5)
    assert var_6 is False

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'simple_string'
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == 'simple_string'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = module_1.render_variable(var_0, var_6, var_5)
    assert var_7 == 'My Project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = "{{ cookiecutter.project_name.replace(' ', '_') }}"
    var_7 = module_1.render_variable(var_0, var_6, var_5)
    assert var_7 == 'My_Project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 123
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == '123'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.name }}'
    var_7 = 'static'
    var_8 = [var_6, var_7]
    var_9 = module_1.render_variable(var_0, var_8, var_5)
    var_10 = bool(var_9 == ['test', 'static'])
    assert var_10 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'user'
    var_3 = 'admin'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'key_template'
    var_7 = 'static_key'
    var_8 = '{{ cookiecutter.user }}'
    var_9 = 'static_val'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_1.render_variable(var_0, var_10, var_5)
    var_12 = bool(var_11 == {'key_template': 'admin', 'static_key': 'static_val'})
    assert var_12 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'base'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'outer'
    var_7 = 'inner'
    var_8 = '{{ cookiecutter.base }}'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = {var_6: var_10}
    var_12 = module_1.render_variable(var_0, var_11, var_5)
    var_13 = bool(var_12 == [{'outer': [{'inner': 'val'}]}])
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_prompt_choice_for_config_with_input_calls_read_user_choice. Retrieved 15/18 statements.


import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = 'other'
    var_6 = [var_4, var_5]
    var_7 = 'key'
    var_8 = True
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'my_project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = []
    var_3 = 'key'
    var_4 = True
    var_5 = module_1.prompt_choice_for_config(var_1, var_0, var_3, var_2, var_4)

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = 'other'
    var_6 = [var_4, var_5]
    var_7 = 'key'
    var_8 = False
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'other'
    var_10 = 'my_project'
    var_11 = 'other'
    var_12 = [var_10, var_11]
    var_13 = None
    var_14 = ''

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}_repo'
    var_5 = '{{ cookiecutter.project_name }}_app'
    var_6 = [var_4, var_5]
    var_7 = 'key'
    var_8 = True
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'test_repo'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = '__prompt__'
    var_8 = 'Custom Prompt'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = module_1.prompt_choice_for_config(var_3, var_0, var_6, var_5, var_11, var_10)
    assert var_12 == 'test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_process_json_preserves_order. Retrieved 2/6 statements.


import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{"key": "value", "id": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'id'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.OrderedDict(*var_8, **var_9)
    var_11 = module_1.process_json(var_0)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", invalid}'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 8/9 statements.
# Partially parsed test_prompt_for_config_with_input. Retrieved 17/18 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = '_internal'
    var_4 = 'My Project'
    var_5 = '{{ cookiecutter.project_name.replace(" ", "_").lower() }}'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['project_name']
    assert var_11 == 'My Project'
    var_12 = var_10['repo_name']
    assert var_12 == 'my_project'
    var_13 = var_10['_internal']
    assert var_13 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'license'
    var_2 = 'MIT'
    var_3 = 'Apache'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['license']
    assert var_9 == 'MIT'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'config_dict'
    var_3 = 'Test'
    var_4 = 'key'
    var_5 = '{{ cookiecutter.project_name }}'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['config_dict']
    var_12 = bool(var_10['config_dict'] == {'key': 'Test'})
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'My Project'
    var_1 = True
    var_2 = '1'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'use_git'
    var_6 = 'license'
    var_7 = '_internal'
    var_8 = 'default_name'
    var_9 = 'MIT'
    var_10 = 'Apache'
    var_11 = [var_9, var_10]
    var_12 = 'hidden'
    var_13 = {var_4: var_8, var_5: var_1, var_6: var_11, var_7: var_12}
    var_14 = {var_3: var_13}
    var_15 = False
    var_16 = module_0.prompt_for_config(var_14, var_15)
    var_17 = var_16['project_name']
    assert var_17 == 'My Project'
    var_18 = var_16['use_git']
    assert var_18 is True
    var_19 = var_16['license']
    assert var_19 == 'MIT'
    var_20 = var_16['_internal']
    assert var_20 == 'hidden'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken'
    var_3 = 'test'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'default_val'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'default'
    var_5 = module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'user_input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'simple_var'
    var_1 = 'default'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'default'
    var_2 = 'PROMPT: '
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'default'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'valid_input'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 9/11 statements.
# Partially parsed test_prompt_for_config_with_rendering. Retrieved 9/14 statements.
# Partially parsed test_prompt_for_config_handles_dicts. Retrieved 11/13 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'my_project'
    var_10 = var_8['version']
    assert var_10 == '1.0.0'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = 'my_project'
    var_4 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['repo_name']
    assert var_9 == 'my_project'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.prompt_for_config(var_4, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_internal_var'
    var_2 = 'public_var'
    var_3 = 'secret'
    var_4 = 'visible'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['_internal_var']
    assert var_9 == 'secret'
    var_10 = var_8['public_var']
    assert var_10 == 'visible'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = 'author'
    var_3 = 'license'
    var_4 = 'admin'
    var_5 = 'MIT'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['metadata']
    var_12 = bool(var_10['metadata'] == {'author': 'admin', 'license': 'MIT'})
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__prompts__'
    var_3 = 'my_project'
    var_4 = 'Enter your project name:'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)
    var_10 = 'project_name'
    var_11 = bool('project_name' in var_9)
    assert var_11 is True
    var_12 = '__prompts__'
    var_13 = bool('__prompts__' not in var_7['cookiecutter'])
    assert var_13 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 10/11 statements.
# Partially parsed test_prompt_for_config_no_input_with_list_options. Retrieved 10/11 statements.
# Partially parsed test_prompt_for_config_with_no_input_and_complex_dict. Retrieved 10/11 statements.
# Partially parsed test_prompt_for_config_with_no_input_boolean. Retrieved 5/6 statements.
# Partially parsed test_prompt_for_config_full_interaction. Retrieved 17/19 statements.
# Partially parsed test_prompt_for_config_with_user_input. Retrieved 13/14 statements.
# Partially parsed test_prompt_for_config_with_prompts_dict. Retrieved 12/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal_var'
    var_3 = '__meta__'
    var_4 = 'my_project'
    var_5 = 'hidden'
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = '_unused'
    var_3 = 'web'
    var_4 = 'api'
    var_5 = [var_3, var_4]
    var_6 = 123
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'base'
    var_2 = 'nested'
    var_3 = 'val'
    var_4 = 'key'
    var_5 = '{{ cookiecutter.base }}'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken'
    var_3 = 'my_project'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'use_docker'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'user_key'
    var_1 = 'user_val'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'choice'
    var_5 = 'config_dict'
    var_6 = 'default'
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_7, var_8]
    var_10 = 'sub'
    var_11 = 'val'
    var_12 = {var_10: var_11}
    var_13 = {var_3: var_6, var_4: var_9, var_5: var_12}
    var_14 = {var_2: var_13}
    var_15 = True
    var_16 = module_0.prompt_for_config(var_14, var_15)
    var_17 = var_16['name']
    assert var_17 == 'default'
    var_18 = var_16['choice']
    assert var_18 == 'a'
    var_19 = var_16['config_dict']
    var_20 = bool(var_16['config_dict'] == {'sub': 'val'})
    assert var_20 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'new_name'
    var_1 = '2'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'choice'
    var_5 = 'default'
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = False
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['name']
    assert var_13 == 'new_name'
    var_14 = var_12['choice']
    assert var_14 == 'b'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = '__prompts__'
    var_3 = '_secret'
    var_4 = 'default'
    var_5 = 'Enter your name: '
    var_6 = {var_1: var_5}
    var_7 = 'hidden'
    var_8 = {var_1: var_4, var_2: var_6, var_3: var_7}
    var_9 = {var_0: var_8}
    var_10 = False
    var_11 = module_0.prompt_for_config(var_9, var_10)
    var_12 = var_11['name']
    assert var_12 == 'custom_val'
    var_13 = var_11['_secret']
    assert var_13 == 'hidden'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_variables. Retrieved 9/10 statements.
# Partially parsed test_prompt_for_config_preserves_order. Retrieved 9/11 statements.
# Partially parsed test_prompt_for_config_handles_prefix. Retrieved 9/13 statements.
# Partially parsed test_prompt_for_config_with_template_errors. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal_var'
    var_3 = '__rendered_var__'
    var_4 = 'my_project'
    var_5 = 'secret'
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'use_feature'
    var_3 = 'options_list'
    var_4 = 'default_name'
    var_5 = True
    var_6 = 'opt1'
    var_7 = 'opt2'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_4, var_2: var_5, var_3: var_8}
    var_10 = {var_0: var_9}
    var_11 = False
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['project_name']
    assert var_13 == 'new_name'
    var_14 = var_12['use_feature']
    assert var_14 is False
    var_15 = var_12['options_list']
    assert var_15 == 'opt2'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken_var'
    var_3 = 'name'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'base'
    var_2 = 'nested_dict'
    var_3 = 'value'
    var_4 = 'key'
    var_5 = '{{ cookiecutter.base }}'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['nested_dict']['key']
    assert var_11 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '__prompts__'
    var_2 = 'var_name'
    var_3 = '_ext'
    var_4 = 'default'
    var_5 = 'val'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Custom Prompt Message'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_6, var_1: var_8}
    var_10 = False
    var_11 = module_0.prompt_for_config(var_9, var_10)
    var_12 = bool(True)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.prompt_for_config(var_4, var_2)
    var_6 = var_5['is_enabled']
    assert var_6 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'z'
    var_2 = 'a'
    var_3 = 'last'
    var_4 = 'first'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_hidden'
    var_2 = 'visible'
    var_3 = 'secret'
    var_4 = 'hello'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = '_hidden'
    var_10 = bool('_hidden' in var_8)
    assert var_10 is True
    var_11 = var_8['_hidden']
    assert var_11 == 'secret'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'root'
    var_2 = 'data'
    var_3 = 'base'
    var_4 = 'sub'
    var_5 = '{{ cookiecutter.root }}'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['data']['sub']
    assert var_11 == 'base'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_list'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)
    var_10 = var_9['my_list']
    assert var_10 == 'b'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'slug'
    var_3 = 'Project'
    var_4 = '{{ cookiecutter.name.lower().replace(" ", "_") }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['slug']
    assert var_9 == 'project'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'var'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = 2
    var_8 = '  [dim][1/1][/]'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'var'
    var_2 = '{{ undefined_variable }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'Undefined'
    var_6 = True
    var_7 = module_0.prompt_for_config(var_4, var_6)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.prompt_for_config(var_4, var_2)
    var_6 = var_5['flag']
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_user_choice_with_string_prompt. Retrieved 8/10 statements.
# Partially parsed test_read_user_choice_with_dict_prompt_and_custom_labels. Retrieved 14/16 statements.
# Partially parsed test_read_user_choice_with_prefix. Retrieved 8/12 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.read_user_choice(var_0, var_1)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'fruit'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'apple'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'color'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'green'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'yes'
    var_1 = 'no'
    var_2 = [var_0, var_1]
    var_3 = 'decision'
    var_4 = 'Do you want to continue?'
    var_5 = {var_3: var_4}
    var_6 = 'decision'
    var_7 = module_0.read_user_choice(var_6, var_2, var_5)
    assert var_7 == 'no'
    var_8 = 'Do you want to continue?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = [var_0, var_1]
    var_3 = 'choice'
    var_4 = '__prompt__'
    var_5 = '1'
    var_6 = '2'
    var_7 = 'Pick an option:'
    var_8 = 'Alpha'
    var_9 = 'Beta'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = 'choice'
    var_13 = module_0.read_user_choice(var_12, var_2, var_11)
    assert var_13 == 'A'
    var_14 = 'Pick an option:'
    var_15 = '[bold]Alpha[/]'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'low'
    var_1 = 'high'
    var_2 = [var_0, var_1]
    var_3 = 'level'
    var_4 = '>>> '
    var_5 = module_0.read_user_choice(var_3, var_2, prefix=var_4)
    assert var_5 == 'high'
    var_6 = 0
    var_7 = '>>> Select level'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 10/11 statements.
# Partially parsed test_prompt_for_config_no_input_with_list_choices. Retrieved 9/10 statements.
# Partially parsed test_prompt_for_config_with_input_interaction. Retrieved 6/8 statements.
# Partially parsed test_prompt_for_config_complex_dict_structure. Retrieved 11/12 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal_var'
    var_3 = '__rendered_var__'
    var_4 = 'my_project'
    var_5 = 'secret'
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = 'web'
    var_3 = 'api'
    var_4 = 'cli'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken_var'
    var_3 = 'my_project'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'settings'
    var_3 = 'my_project'
    var_4 = 'enabled'
    var_5 = 'mode'
    var_6 = True
    var_7 = 'fast'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_1: var_3, var_2: var_8}
    var_10 = {var_0: var_9}

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_hidden'
    var_3 = 'test'
    var_4 = 'val'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'test'
    var_10 = var_8['_hidden']
    assert var_10 == 'val'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_choose_nested_template_new_style_success. Retrieved 17/18 statements.
# Partially parsed test_choose_nested_template_old_style_success. Retrieved 12/13 statements.


import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = 'path'
    var_5 = 'templates/option1'
    var_6 = {var_4: var_5}
    var_7 = 'templates/option2'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '.'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_0.Path(*var_13, **var_14)
    var_16 = var_15.resolve()
    var_17 = True
    var_18 = module_1.choose_nested_template(var_11, var_16, var_17)

import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice1 (templates/old_choice)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '.'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = var_9.resolve()
    var_11 = True
    var_12 = module_1.choose_nested_template(var_5, var_10, var_11)
    var_13 = 'templates/old_choice'

import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = '/absolute/path/to/template'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '.'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)
    var_13 = var_12.resolve()
    var_14 = True
    var_15 = module_1.choose_nested_template(var_8, var_13, var_14)

import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = var_6.resolve()
    var_8 = True
    var_9 = module_1.choose_nested_template(var_2, var_7, var_8)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_choose_nested_template_predicate_is_false_when_template_is_absolute. Retrieved 12/18 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = '/absolute/path/to/template'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp'
    var_10 = module_0.choose_nested_template(var_8, var_9)
    var_11 = str(var_10)
    assert var_11 == 'Illegal template path'



# Parsed testcases at query #16
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_private_dict'
    var_2 = 'public_var'
    var_3 = 'some'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = 'value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = '_private_dict'
    var_12 = bool('_private_dict' not in var_10)
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_prompt_and_delete_deletes_directory_when_ok_to_delete_is_true. Retrieved 9/16 statements.


import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'test_directory_to_delete'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = True
    var_7 = module_1.prompt_and_delete(var_3, var_6)
    assert var_7 is True
    var_8 = var_3.exists()
    var_9 = bool(not var_8)
    assert var_9 is True
    var_10 = '.'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_internal_var'
    var_2 = 'project_name'
    var_3 = 'some_value'
    var_4 = 'my_project'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = '_internal_var'
    var_10 = bool('_internal_var' in var_8)
    assert var_10 is True
    var_11 = var_8['_internal_var']
    assert var_11 == 'some_value'
    var_12 = 'project_name'
    var_13 = bool('project_name' in var_8)
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'simple_string_prompt'
    var_2 = {var_0: var_1}
    var_3 = 'A'
    var_4 = 'B'
    var_5 = [var_3, var_4]
    var_6 = module_0.read_user_choice(var_0, var_5, var_2)
    assert var_6 == 'A'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_read_user_yes_no_uses_var_name_as_question_when_no_prompts. Retrieved 3/5 statements.
# Partially parsed test_read_user_yes_no_uses_prompt_mapping. Retrieved 5/7 statements.
# Partially parsed test_read_user_yes_no_applies_prefix. Retrieved 5/7 statements.
# Partially parsed test_read_user_yes_no_with_prefix_and_prompts. Retrieved 7/9 statements.
# Partially parsed test_read_user_yes_no_handles_empty_prompts_dict. Retrieved 4/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'confirm'
    var_1 = False
    var_2 = module_0.read_user_yes_no(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'delete'
    var_1 = 'Are you sure you want to delete this?'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.read_user_yes_no(var_0, var_3, var_2)
    assert var_4 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = '[INFO] '
    var_3 = module_0.read_user_yes_no(var_0, var_1, prefix=var_2)
    assert var_3 is True
    var_4 = '[INFO] test'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'exit'
    var_1 = 'Exit program?'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = '?'
    var_5 = module_0.read_user_yes_no(var_0, var_3, var_2, var_4)
    assert var_5 is False
    var_6 = '?Exit program?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = {}
    var_3 = module_0.read_user_yes_no(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'true'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 'T'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'YES'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 'y'
    var_10 = var_0.process_response(var_9)
    assert var_10 is True
    var_11 = '  on  '
    var_12 = var_0.process_response(var_11)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '0'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'false'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = 'f'
    var_6 = var_0.process_response(var_5)
    assert var_6 is False
    var_7 = 'no'
    var_8 = var_0.process_response(var_7)
    assert var_8 is False
    var_9 = 'n'
    var_10 = var_0.process_response(var_9)
    assert var_10 is False
    var_11 = 'OFF'
    var_12 = var_0.process_response(var_11)
    assert var_12 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #22
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_private_var'
    var_2 = '__internal_var__'
    var_3 = 'some_value'
    var_4 = 'template_string'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['_private_var']
    assert var_9 == 'some_value'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_prompt_and_delete_skips_deletion_when_user_says_no. Retrieved 3/4 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    var_3 = bool(var_2 is not True)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"'

def test_case_0():
    var_0 = "{'broken': json}"

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key":'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_prompt_and_delete_skips_deletion_when_user_says_no. Retrieved 3/4 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'fake_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    var_3 = bool(var_2 is not True)
    assert var_3 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_process_json_invalid_json_syntax. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_list. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_string. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_number. Retrieved 3/5 statements.


import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.OrderedDict(*var_8, **var_9)
    var_11 = module_1.process_json(var_0)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.OrderedDict(*var_1, **var_2)
    var_4 = module_1.process_json(var_0)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Unable to decode to JSON.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Requires JSON dict.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Requires JSON dict.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Requires JSON dict.'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_read_user_variable_returns_default_on_none. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_uses_prefix. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_uses_var_name_as_fallback. Retrieved 4/6 statements.
# Partially parsed test_read_user_variable_handles_none_prompts. Retrieved 4/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'entered_value'
    var_2 = 'test_var'
    var_3 = 'default'
    var_4 = module_0.read_user_variable(var_2, var_3)
    assert var_4 == 'entered_value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = module_0.read_user_variable(var_0, var_3, var_2)
    assert var_4 == 'response'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default'
    var_2 = 'PRE: '
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'response'
    var_4 = 'PRE: test_var'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default'
    var_2 = {}
    var_3 = module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'response'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default'
    var_2 = None
    var_3 = module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'response'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_prompt_for_config_calls_read_user_variable. Retrieved 7/9 statements.
# Partially parsed test_prompt_for_config_calls_read_user_yes_no. Retrieved 7/9 statements.
# Partially parsed test_prompt_for_config_calls_read_user_dict. Retrieved 13/15 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal_var'
    var_3 = 'my_project'
    var_4 = 'internal'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'my_project'
    var_10 = var_8['_internal_var']
    assert var_10 == 'internal'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = 'My Project'
    var_4 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'My Project'
    var_10 = var_8['repo_name']
    assert var_10 == 'My_Project'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = 'web'
    var_3 = 'mobile'
    var_4 = 'desktop'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)
    var_10 = var_9['type']
    assert var_10 == 'web'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'metadata'
    var_3 = 'App'
    var_4 = 'version'
    var_5 = 'author'
    var_6 = '1.0'
    var_7 = 'Admin'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_1: var_3, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['metadata']
    var_14 = bool(var_12['metadata'] == {'version': '1.0', 'author': 'Admin'})
    assert var_14 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choice'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.prompt_for_config(var_4, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'default_val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = var_6['name']
    assert var_7 == 'user_input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'use_feature'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = var_6['use_feature']
    assert var_7 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'settings'
    var_5 = 'App'
    var_6 = 'theme'
    var_7 = 'dark'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = False
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['settings']
    var_14 = bool(var_12['settings'] == {'key': 'val'})
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_private_var'
    var_2 = '__internal_var__'
    var_3 = 'some_value'
    var_4 = 'template_string'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = '_private_var'
    var_10 = bool('_private_var' in var_8)
    assert var_10 is True
    var_11 = var_8['_private_var']
    assert var_11 == 'some_value'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_read_user_choice_basic_functionality. Retrieved 6/8 statements.
# Partially parsed test_read_user_choice_with_prefix. Retrieved 8/12 statements.
# Partially parsed test_read_user_choice_with_string_prompts. Retrieved 7/9 statements.
# Partially parsed test_read_user_choice_with_dict_prompts_and_custom_labels. Retrieved 13/15 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.read_user_choice(var_0, var_1)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'fruit'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'apple'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'low'
    var_1 = 'high'
    var_2 = [var_0, var_1]
    var_3 = 'level'
    var_4 = '[INFO] '
    var_5 = module_0.read_user_choice(var_3, var_2, prefix=var_4)
    assert var_5 == 'high'
    var_6 = 0
    var_7 = '[INFO] Select level'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'yes'
    var_1 = 'no'
    var_2 = [var_0, var_1]
    var_3 = 'choice'
    var_4 = 'Do you want to continue?'
    var_5 = {var_3: var_4}
    var_6 = module_0.read_user_choice(var_3, var_2, var_5)
    assert var_6 == 'yes'
    var_7 = 'Do you want to continue?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = [var_0, var_1]
    var_3 = 'var'
    var_4 = '__prompt__'
    var_5 = '1'
    var_6 = '2'
    var_7 = 'Custom Prompt'
    var_8 = 'Label One'
    var_9 = 'Label Two'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = {var_3: var_10}
    var_12 = module_0.read_user_choice(var_3, var_2, var_11)
    assert var_12 == 'A'
    var_13 = 'Custom Prompt'
    var_14 = '[bold magenta]1[/] - [bold]Label One[/]'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_process_json_invalid_json_syntax. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_list. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_string. Retrieved 3/5 statements.
# Partially parsed test_process_json_preserves_order. Retrieved 2/4 statements.


import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.OrderedDict(*var_8, **var_9)
    var_11 = module_1.process_json(var_0)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", invalid}'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Unable to decode to JSON.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Requires JSON dict.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Requires JSON dict.'

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.OrderedDict(*var_1, **var_2)
    var_4 = module_1.process_json(var_0)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_process_json_preserves_order. Retrieved 2/7 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "num": 123}'
    var_1 = 'key'
    var_2 = 'num'
    var_3 = 'value'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.process_json(var_0)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = module_0.process_json(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_var'
    var_2 = '{{ undefined_variable }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = "Unable to render variable 'my_var'"

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = "Unable to render variable 'some_key'"



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_prompt_for_config_no_input_skips_dict_prompting. Retrieved 13/18 statements.
# Partially parsed test_prompt_for_config_with_input_calls_dict_prompting. Retrieved 11/16 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '_hidden'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'secret'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'key'
    var_10 = 'updated_value'
    var_11 = True
    var_12 = module_0.prompt_for_config(var_8, var_11)
    var_13 = var_12['my_dict']
    var_14 = bool(var_12['my_dict'] == {'key': 'value'})
    assert var_14 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'key'
    var_8 = 'user_input_value'
    var_9 = False
    var_10 = module_0.prompt_for_config(var_6, var_9)
    var_11 = var_10['my_dict']
    var_12 = bool(var_10['my_dict'] == {'key': 'user_input_value'})
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_read_user_yes_no_uses_var_name_as_question_when_no_prompts. Retrieved 6/8 statements.
# Partially parsed test_read_user_yes_no_uses_prompt_mapping_when_available. Retrieved 8/10 statements.
# Partially parsed test_read_user_yes_no_applies_prefix_correctly. Retrieved 6/8 statements.
# Partially parsed test_read_user_yes_no_handles_empty_prompts_dict. Retrieved 5/7 statements.
# Partially parsed test_read_user_yes_no_handles_missing_key_in_prompts. Retrieved 7/9 statements.
# Partially parsed test_read_user_yes_no_handles_none_value_in_prompts. Retrieved 7/9 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'confirm'
    var_1 = False
    var_2 = None
    var_3 = '[?] '
    var_4 = module_0.read_user_yes_no(var_0, var_1, var_2, var_3)
    assert var_4 is True
    var_5 = '[?] confirm'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'action'
    var_1 = 'Do you want to proceed?'
    var_2 = {var_0: var_1}
    var_3 = 'action'
    var_4 = True
    var_5 = ''
    var_6 = module_0.read_user_yes_no(var_3, var_4, var_2, var_5)
    assert var_6 is False
    var_7 = 'Do you want to proceed?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = None
    var_3 = 'PROMPT: '
    var_4 = module_0.read_user_yes_no(var_0, var_1, var_2, var_3)
    var_5 = 'PROMPT: test'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = {}
    var_3 = ''
    var_4 = module_0.read_user_yes_no(var_0, var_1, var_2, var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'other'
    var_1 = 'Something else'
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = False
    var_5 = ''
    var_6 = module_0.read_user_yes_no(var_3, var_4, var_2, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = False
    var_5 = ''
    var_6 = module_0.read_user_yes_no(var_3, var_4, var_2, var_5)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'missing_key'
    var_1 = 'default'
    var_2 = 'other_key'
    var_3 = 'some_prompt'
    var_4 = {var_2: var_3}
    var_5 = module_0.read_user_variable(var_0, var_1, var_4)
    assert var_5 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'any_key'
    var_1 = 'default'
    var_2 = {}
    var_3 = module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default'
    var_2 = ''
    var_3 = {var_0: var_2}
    var_4 = module_0.read_user_variable(var_0, var_1, var_3)
    assert var_4 == 'value'



# Parsed testcases at query #12
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '__prompts__'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['my_dict']
    var_12 = bool(var_10['my_dict'] == {'key': 'updated_value'})
    assert var_12 is True



# Parsed testcases at query #13
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_choice'
    var_2 = '_private_var'
    var_3 = 'option1'
    var_4 = 'option2'
    var_5 = [var_3, var_4]
    var_6 = 'secret'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['my_choice']
    assert var_11 == 'option1'
    var_12 = var_10['_private_var']
    assert var_12 == 'secret'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_choose_nested_template_new_style_success. Retrieved 17/22 statements.
# Partially parsed test_choose_nested_template_old_style_success. Retrieved 18/23 statements.


import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = 'subdir/template1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = '/tmp/repo'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)
    var_14 = '/tmp/repo/subdir/template1'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_0.Path(*var_15, **var_16)
    var_18 = module_1.choose_nested_template(var_8, var_9)
    var_19 = [var_14]
    var_20 = {}
    var_21 = module_0.Path(*var_19, **var_20)
    var_22 = str(var_21)
    var_23 = bool(var_18 == var_22)
    assert var_23 is True

import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice1 (path/to/template)'
    var_3 = 'choice2 (other/path)'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = '/tmp/repo'
    var_8 = '/tmp/repo'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)
    var_12 = var_11.resolve()
    var_13 = '/tmp/repo/path/to/template'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Path(*var_14, **var_15)
    var_17 = var_16.resolve()
    var_18 = module_1.choose_nested_template(var_6, var_7)
    var_19 = [var_13]
    var_20 = {}
    var_21 = module_0.Path(*var_19, **var_20)
    var_22 = var_21.resolve()
    var_23 = str(var_22)
    var_24 = bool(var_18 == var_23)
    assert var_24 is True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = '/absolute/path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = None
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 is None

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = True
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 is True
    var_4 = False
    var_5 = {}
    var_6 = module_1.render_variable(var_0, var_4, var_5)
    assert var_6 is False

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'simple_string'
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == 'simple_string'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = module_1.render_variable(var_0, var_6, var_5)
    assert var_7 == 'my_project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Peanut Butter Cookie'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = "{{ cookiecutter.project_name.replace(' ', '_') }}"
    var_7 = module_1.render_variable(var_0, var_6, var_5)
    assert var_7 == 'Peanut_Butter_Cookie'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.name }}'
    var_7 = 'static'
    var_8 = [var_6, var_7]
    var_9 = module_1.render_variable(var_0, var_8, var_5)
    var_10 = bool(var_9 == ['test', 'static'])
    assert var_10 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'cookiecutter'
    var_2 = 'user'
    var_3 = 'admin'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'key_template'
    var_7 = 'static_key'
    var_8 = '{{ cookiecutter.user }}'
    var_9 = 'value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_6: var_3, var_7: var_9}
    var_12 = module_1.render_variable(var_0, var_10, var_5)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 123
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == '123'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_process_response_false_values. Retrieved 14/15 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'yes'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'y'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 'true'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = '1'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 't'
    var_10 = var_0.process_response(var_9)
    assert var_10 is True
    var_11 = 'on'
    var_12 = var_0.process_response(var_11)
    assert var_12 is True
    var_13 = '  YES  '
    var_14 = var_0.process_response(var_13)
    assert var_14 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'no'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'n'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = 'false'
    var_6 = var_0.process_response(var_5)
    assert var_6 is False
    var_7 = '0'
    var_8 = 'f'
    var_9 = var_0.process_response(var_8)
    assert var_9 is False
    var_10 = 'off'
    var_11 = var_0.process_response(var_10)
    assert var_11 is False
    var_12 = '  NO  '
    var_13 = var_0.process_response(var_12)
    assert var_13 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)
    var_3 = ''
    var_4 = var_0.process_response(var_3)



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_read_user_variable_returns_default_when_input_is_none.
# Partially parsed test_read_user_variable_uses_var_name_as_question. Retrieved 3/5 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_applies_prefix. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_handles_complex_prompt_and_prefix. Retrieved 7/9 statements.
# Partially parsed test_read_user_variable_retries_on_none. Retrieved 5/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'admin'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'Enter your name: '
    var_2 = {var_0: var_1}
    var_3 = 'admin'
    var_4 = module_0.read_user_variable(var_0, var_3, var_2)
    assert var_4 == 'value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'admin'
    var_2 = '[INFO] '
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'value'
    var_4 = '[INFO] username'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'Name?'
    var_2 = {var_0: var_1}
    var_3 = 'admin'
    var_4 = '> '
    var_5 = module_0.read_user_variable(var_0, var_3, var_2, var_4)
    assert var_5 == 'value'
    var_6 = '> Name?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'recovered'
    var_2 = 'username'
    var_3 = 'admin'
    var_4 = module_0.read_user_variable(var_2, var_3)
    assert var_4 == 'recovered'



# Parsed testcases at query #18
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '__prompts__'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['my_dict']
    var_12 = bool(var_10['my_dict'] == {'key': 'new_value'})
    assert var_12 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_prompt_for_config_handles_list_as_choice. Retrieved 13/19 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_choice'
    var_2 = '_some_private_var'
    var_3 = 'option1'
    var_4 = 'option2'
    var_5 = [var_3, var_4]
    var_6 = 'value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = 'my_choice'
    var_12 = var_10[var_11]
    var_13 = bool(var_3)
    assert var_13 is True
    var_14 = var_10['my_choice']
    assert var_14 == 'option1'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_true_is_dir. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_no_input_true_is_file. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_yes. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_no_then_exit. Retrieved 3/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/file.zip'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_dir. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_no_input_file. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_yes. Retrieved 3/6 statements.
# Partially parsed test_prompt_and_delete_user_says_no_then_reuse. Retrieved 4/5 statements.
# Partially parsed test_prompt_and_delete_user_says_no_then_exit. Retrieved 3/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/file'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = '/fake/path'
    var_3 = module_0.prompt_and_delete(var_2, var_0)
    assert var_3 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = False
    var_1 = '/fake/path'
    var_2 = module_0.prompt_and_delete(var_1, var_0)
    assert var_2 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_prompt_for_config_no_input_skips_dict_user_input. Retrieved 11/16 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '_private'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'hidden'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['my_dict']
    var_12 = bool(var_10['my_dict'] == {'key': 'updated_value'})
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------




import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.OrderedDict(*var_8, **var_9)
    var_11 = module_1.process_json(var_0)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", invalid}'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = []
    var_2 = {}
    var_3 = module_0.OrderedDict(*var_1, **var_2)
    var_4 = module_1.process_json(var_0)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_choice'
    var_2 = '_internal'
    var_3 = 'option1'
    var_4 = 'option2'
    var_5 = [var_3, var_4]
    var_6 = 'hidden'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['my_choice']
    assert var_11 == 'option1'



# Parsed testcases at query #25
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '_private_dict'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'hidden'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = False
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['my_dict']
    var_14 = bool(var_12['my_dict'] == {'key': 'updated'})
    assert var_14 is True



