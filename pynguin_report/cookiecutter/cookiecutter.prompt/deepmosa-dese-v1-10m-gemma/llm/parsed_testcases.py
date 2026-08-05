####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_process_json_valid_dict. Retrieved 9/10 statements.
# Partially parsed test_process_json_invalid_json_syntax. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_list. Retrieved 3/5 statements.
# Partially parsed test_process_json_not_a_dict_string. Retrieved 3/5 statements.
# Partially parsed test_process_json_preserves_order. Retrieved 2/4 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.process_json(var_0)

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.OrderedDict()
    var_2 = module_1.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value",}'
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
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_process_json_valid_dict. Retrieved 9/10 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.process_json(var_0)

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.OrderedDict()
    var_2 = module_1.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value",}'
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
    var_0 = '123'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_user_dict_uses_prompt_when_available. Retrieved 11/15 statements.
# Partially parsed test_read_user_dict_uses_var_name_when_no_prompts. Retrieved 10/14 statements.
# Partially parsed test_read_user_dict_uses_var_name_when_prompts_missing_key. Retrieved 9/13 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'not a dict'
    var_2 = module_0.read_user_dict(var_0, var_1)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'user_data'
    var_3 = 'Enter your info'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = 'PROMPT: '
    var_7 = module_0.read_user_dict(var_2, var_5, var_4, var_6)
    var_8 = 'PROMPT:Enter your info [cyan bold](DEFAULT_DISPLAY)[/]'
    var_9 = {}
    var_10 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'username'
    var_3 = 'default'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = module_0.read_user_dict(var_2, var_5)
    var_7 = 'username [cyan bold](DEFAULT_DISPLAY)[/]'
    var_8 = {var_3: var_4}
    var_9 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'Ignore me'
    var_2 = {var_0: var_1}
    var_3 = 'target_key'
    var_4 = {}
    var_5 = module_0.read_user_dict(var_3, var_4, var_2)
    var_6 = 'target_key [cyan bold](DEFAULT_DISPLAY)[/]'
    var_7 = {}
    var_8 = False



# Parsed testcases at query #4
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
    var_7 = 'some_key'
    var_8 = True
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'my_project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = []
    var_3 = 'some_key'
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
    var_7 = 'some_key'
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
    var_1 = 'user'
    var_2 = 'admin'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.user }}_repo'
    var_5 = 'guest_repo'
    var_6 = [var_4, var_5]
    var_7 = 'repo_name'
    var_8 = True
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'admin_repo'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_read_user_variable_returns_default_when_no_input. Retrieved 3/13 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt. Retrieved 7/16 statements.
# Partially parsed test_read_user_variable_applies_prefix. Retrieved 7/13 statements.
# Partially parsed test_read_user_variable_handles_none_input_retry. Retrieved 6/10 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'test_var'
    var_2 = module_0.read_user_variable(var_1, var_0)
    assert var_2 == 'default_val'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'user_input'
    var_1 = 'test_var'
    var_2 = 'Custom Question'
    var_3 = {var_1: var_2}
    var_4 = 'default_val'
    var_5 = module_0.read_user_variable(var_1, var_4, var_3)
    assert var_5 == 'user_input'
    var_6 = 'Prompt'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'input'
    var_1 = 'test_var'
    var_2 = 'default_val'
    var_3 = 'PRE_: '
    var_4 = module_0.read_user_variable(var_1, var_2, prefix=var_3)
    assert var_4 == 'input'
    var_5 = 'Prompt'
    var_6 = 'PRE_: test_var'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'valid_value'
    var_2 = [var_0, var_1]
    var_3 = 'test_var'
    var_4 = 'default_val'
    var_5 = module_0.read_user_variable(var_3, var_4)
    assert var_5 == 'valid_value'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_read_user_variable_returns_default_when_no_input. Retrieved 3/7 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt. Retrieved 7/11 statements.
# Partially parsed test_read_user_variable_uses_prefix. Retrieved 5/9 statements.
# Partially parsed test_read_user_variable_retries_on_none. Retrieved 5/8 statements.
# Partially parsed test_read_user_variable_handles_empty_prompts_dict. Retrieved 4/8 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'some_value'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'Custom Prompt Text'
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'default_val'
    var_5 = module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'input_val'
    var_6 = 'Custom Prompt Text'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default_val'
    var_2 = 'PRE_'
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'input_val'
    var_4 = 'PRE_test_var'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'valid_input'
    var_2 = 'test_var'
    var_3 = 'default_val'
    var_4 = module_0.read_user_variable(var_2, var_3)
    assert var_4 == 'valid_input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default_val'
    var_2 = {}
    var_3 = module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'val'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_read_user_yes_no_uses_var_name_when_no_prompts. Retrieved 3/5 statements.
# Partially parsed test_read_user_yes_no_uses_prompt_from_dict. Retrieved 5/7 statements.
# Partially parsed test_read_user_yes_no_applies_prefix. Retrieved 5/7 statements.
# Partially parsed test_read_user_yes_no_with_prefix_and_prompts. Retrieved 7/9 statements.
# Partially parsed test_read_user_yes_no_handles_empty_prompts_dict. Retrieved 4/6 statements.
# Partially parsed test_read_user_yes_no_handles_none_prompts. Retrieved 4/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'confirm'
    var_1 = False
    var_2 = module_0.read_user_yes_no(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'confirm'
    var_1 = 'Do you want to proceed?'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.read_user_yes_no(var_0, var_3, var_2)
    assert var_4 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = '[PROMPT] '
    var_3 = module_0.read_user_yes_no(var_0, var_1, prefix=var_2)
    assert var_3 is True
    var_4 = '[PROMPT] test'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'action'
    var_1 = 'Delete file?'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = '?'
    var_5 = module_0.read_user_yes_no(var_0, var_3, var_2, var_4)
    assert var_5 is False
    var_6 = '?Delete file?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = {}
    var_3 = module_0.read_user_yes_no(var_0, var_1, var_2)
    assert var_3 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None
    var_3 = module_0.read_user_yes_no(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_process_response_invalid_value. Retrieved 3/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'true'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 't'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'yes'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = 'y'
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
    var_11 = 'off'
    var_12 = var_0.process_response(var_11)
    assert var_12 is False
    var_13 = '  no  '
    var_14 = var_0.process_response(var_13)
    assert var_14 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_read_user_yes_no_predicate_false_due_to_missing_key. Retrieved 7/11 statements.
# Partially parsed test_read_user_yes_no_predicate_false_due_to_falsy_prompt_value. Retrieved 7/11 statements.
# Partially parsed test_read_user_yes_no_predicate_false_due_to_none_prompts. Retrieved 5/9 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'some_prompt'
    var_2 = {var_0: var_1}
    var_3 = 'target_key'
    var_4 = True
    var_5 = module_0.read_user_yes_no(var_3, var_4, var_2)
    var_6 = 'target_key'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = 'var_name'
    var_4 = True
    var_5 = module_0.read_user_yes_no(var_3, var_4, var_2)
    var_6 = 'var_name'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'any_key'
    var_1 = False
    var_2 = None
    var_3 = module_0.read_user_yes_no(var_0, var_1, var_2)
    var_4 = 'any_key'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_prompt_for_config_no_input. Retrieved 11/12 statements.
# Partially parsed test_prompt_for_config_with_prompts_dict. Retrieved 10/13 statements.
# Partially parsed test_prompt_for_config_empty_list_error. Retrieved 8/11 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_private_var'
    var_3 = '__rendered_var__'
    var_4 = 'my_project'
    var_5 = 'hidden'
    var_6 = '{{ cookiecutter.project_name }}'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

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
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'use_git'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.prompt_for_config(var_4, var_2)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'settings'
    var_2 = 'debug_mode'
    var_3 = 'debug'
    var_4 = '{{ cookiecutter.debug_mode }}'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'user_name'
    var_2 = 'default_val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__prompts__'
    var_3 = 'template'
    var_4 = 'Enter your project name:'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choices'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = 'The list of empty choices'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_dir_exists. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_no_input_file_exists. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_yes_to_delete. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_no_to_delete_and_yes_to_reuse. Retrieved 4/6 statements.
# Partially parsed test_prompt_and_delete_user_says_no_to_delete_and_no_to_reuse_exits. Retrieved 3/6 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 9/14 statements.
# Partially parsed test_prompt_for_config_with_private_vars. Retrieved 9/12 statements.
# Partially parsed test_prompt_for_config_raises_undefined_error. Retrieved 5/9 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'my_project'
    var_4 = '0.1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_internal_id'
    var_2 = 'public_var'
    var_3 = '123'
    var_4 = 'hello'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'broken'
    var_2 = '{{ non_existent }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #13
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.read_user_choice(var_0, var_1)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = 'red'
    var_2 = 'blue'
    var_3 = 'green'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.read_user_choice(var_0, var_4)
    assert var_5 == 'red'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'size'
    var_1 = 'small'
    var_2 = 'large'
    var_3 = [var_1, var_2]
    var_4 = '[INFO] '
    var_5 = module_0.read_user_choice(var_0, var_3, prefix=var_4)
    assert var_5 == 'large'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = 'What color do you want?'
    var_2 = {var_0: var_1}
    var_3 = 'color'
    var_4 = 'red'
    var_5 = 'blue'
    var_6 = [var_4, var_5]
    var_7 = module_0.read_user_choice(var_3, var_6, var_2)
    assert var_7 == 'blue'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = '__prompt__'
    var_2 = '1'
    var_3 = '2'
    var_4 = 'Pick a hue:'
    var_5 = 'Crimson'
    var_6 = 'Azure'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'red'
    var_10 = 'blue'
    var_11 = [var_9, var_10]
    var_12 = 'color'
    var_13 = module_0.read_user_choice(var_12, var_11, var_8)
    assert var_13 == 'red'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'color'
    var_1 = '__prompt__'
    var_2 = 'other'
    var_3 = 'Pick a hue:'
    var_4 = 'something'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'red'
    var_8 = 'blue'
    var_9 = [var_7, var_8]
    var_10 = 'color'
    var_11 = module_0.read_user_choice(var_10, var_9, var_6)
    assert var_11 == 'blue'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_choose_nested_template_new_style_success. Retrieved 15/21 statements.
# Partially parsed test_choose_nested_template_old_style_success. Retrieved 13/19 statements.
# Partially parsed test_choose_nested_template_raises_value_error_on_absolute_path. Retrieved 13/17 statements.
# Partially parsed test_choose_nested_template_raises_value_error_on_none_template. Retrieved 13/16 statements.
# Partially parsed test_choose_nested_template_removes_prompts_from_context. Retrieved 14/18 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 'template1'
    var_4 = 'path'
    var_5 = 'subdir/template1'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = '/tmp/repo'
    var_12 = module_0.choose_nested_template(var_10, var_11)
    var_13 = '/tmp/repo/subdir/template1'
    var_14 = str(var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = '__prompts__'
    var_3 = 'choice1 (path/to/template)'
    var_4 = 'choice2 (other/path)'
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = module_0.choose_nested_template(var_8, var_9)
    var_11 = '/tmp/repo/path/to/template'
    var_12 = str(var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 'template1'
    var_4 = 'path'
    var_5 = '/absolute/path/template1'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = '/tmp/repo'
    var_12 = module_0.choose_nested_template(var_10, var_11)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 'template1'
    var_4 = 'path'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = '/tmp/repo'
    var_12 = module_0.choose_nested_template(var_10, var_11)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 't1'
    var_4 = 'path'
    var_5 = {var_4: var_3}
    var_6 = {var_3: var_5}
    var_7 = 'some_key'
    var_8 = 'some_val'
    var_9 = {var_7: var_8}
    var_10 = {var_1: var_6, var_2: var_9}
    var_11 = {var_0: var_10}
    var_12 = '/tmp/repo'
    var_13 = module_0.choose_nested_template(var_11, var_12)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_process_json_valid_dict. Retrieved 9/10 statements.
# Partially parsed test_process_json_preserves_order. Retrieved 2/4 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", broken_json}'
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
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_process_json_with_valid_dict_string. Retrieved 2/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_prompt_and_delete_evaluates_true_when_no_input_is_true. Retrieved 3/9 statements.
# Partially parsed test_prompt_and_delete_evaluates_true_when_user_says_yes. Retrieved 3/8 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/file.zip'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_true_dir. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_no_input_true_file. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_yes_to_delete. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_user_says_no_to_delete_and_no_to_reuse. Retrieved 3/5 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_vars. Retrieved 8/9 statements.
# Partially parsed test_prompt_for_config_no_input_with_rendering. Retrieved 10/11 statements.
# Partially parsed test_prompt_for_config_basic_execution. Retrieved 10/15 statements.
# Partially parsed test_prompt_for_config_no_input_mode. Retrieved 9/13 statements.
# Partially parsed test_prompt_for_config_with_templates. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_version'
    var_3 = 'my_project'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = '_version'
    var_4 = 'my_project'
    var_5 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_6 = '1.0.0'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken_var'
    var_3 = 'my_project'
    var_4 = '{{ non_existent_variable }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'user_input'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_internal'
    var_4 = 'default_name'
    var_5 = 'hidden'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal'
    var_3 = 'original'
    var_4 = 'hidden'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'template_raw'
    var_1 = 'processed_value'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_internal'
    var_5 = 'hidden'
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_prompt_and_delete_deletes_directory_when_ok. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'test_directory_to_delete'
    var_1 = True
    var_2 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_process_json_valid_dict. Retrieved 9/10 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.process_json(var_0)

import collections as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.OrderedDict()
    var_2 = module_1.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value",}'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '"just a string"'
    var_1 = module_0.process_json(var_0)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.process_json(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_prompt_and_delete_evaluates_true_when_no_input_is_true. Retrieved 3/5 statements.
# Partially parsed test_prompt_and_delete_evaluates_true_when_user_says_yes. Retrieved 3/5 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '/fake/file.zip'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_prompt_for_config_predicate_true. Retrieved 15/19 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '__not_a_user_dict__'
    var_3 = 'inner_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'hidden'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = 'inner_key'
    var_12 = 'updated_value'
    var_13 = False
    var_14 = module_0.prompt_for_config(var_10, var_13)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_prompt_for_config_no_input. Retrieved 11/12 statements.
# Partially parsed test_prompt_for_config_with_user_input. Retrieved 11/12 statements.
# Partially parsed test_prompt_for_config_boolean_input. Retrieved 7/8 statements.
# Partially parsed test_prompt_for_config_dict_input. Retrieved 11/12 statements.
# Partially parsed test_prompt_for_config_empty_list_raises_error. Retrieved 9/14 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_version'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = '1.0.0'
    var_6 = 'author_name'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = '_internal'
    var_4 = 'my_project'
    var_5 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_6 = 'secret'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'license'
    var_2 = '_unused'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'data'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.prompt_for_config(var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = '_meta'
    var_3 = 'version'
    var_4 = 'owner'
    var_5 = '1.0'
    var_6 = 'admin'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'info'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.prompt_for_config(var_10, var_11)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'new_project'
    var_1 = 'John Doe'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'default_project'
    var_6 = 'default_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'use_git'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = 'cookiecutter'
    var_3 = 'settings'
    var_4 = 'default'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choices'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = 'choices'
    var_7 = []
    var_8 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_prompt_and_delete_skips_deletion_when_user_says_no. Retrieved 3/4 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_prompt_and_delete_evaluates_ok_to_delete_false. Retrieved 3/9 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = False
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is None



# Parsed testcases at query #28
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'default'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'some_value'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_choose_nested_template_new_style_success. Retrieved 18/29 statements.
# Partially parsed test_choose_nested_template_old_style_success. Retrieved 12/24 statements.
# Partially parsed test_choose_nested_template_error_on_absolute_path. Retrieved 12/16 statements.
# Partially parsed test_choose_nested_template_error_on_none_path. Retrieved 12/15 statements.
# Partially parsed test_choose_nested_template_handles_prompts_removal. Retrieved 17/26 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'template2'
    var_4 = 'path'
    var_5 = 'template1_dir'
    var_6 = {var_4: var_5}
    var_7 = 'template2_dir'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '/tmp/repo'
    var_13 = '/tmp/repo'
    var_14 = '/tmp/repo/template1_dir'
    var_15 = True
    var_16 = module_0.choose_nested_template(var_11, var_12, var_15)
    var_17 = str(var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'template_one (path/to/template)'
    var_3 = 'template_two (other/path)'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = '/tmp/repo'
    var_8 = '/tmp/repo'
    var_9 = '/tmp/repo/path/to/template'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_6, var_7, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = '/absolute/path/to/template'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'template1'
    var_3 = 'path'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = '__prompts__'
    var_3 = 't1'
    var_4 = 'path'
    var_5 = {var_4: var_3}
    var_6 = {var_3: var_5}
    var_7 = 'some_key'
    var_8 = 'some_val'
    var_9 = {var_7: var_8}
    var_10 = {var_1: var_6, var_2: var_9}
    var_11 = {var_0: var_10}
    var_12 = '/tmp/repo'
    var_13 = '/tmp/repo'
    var_14 = '/tmp/repo/t1'
    var_15 = True
    var_16 = module_0.choose_nested_template(var_11, var_12, var_15)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_choose_nested_template_validates_relative_path. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice1'
    var_3 = 'path'
    var_4 = 'subdir/template_a'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '.'
    var_10 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_choose_nested_template_valid_path_with_no_config. Retrieved 9/11 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice1'
    var_3 = 'path'
    var_4 = 'subdir/template_a'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)
    assert var_11 == '/tmp/repo/subdir/template_a'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice (relative/path)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)
    assert var_8 == '/tmp/repo/relative/path'



