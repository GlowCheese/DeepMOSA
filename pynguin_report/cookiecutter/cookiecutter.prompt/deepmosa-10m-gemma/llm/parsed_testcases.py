####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_dir_exists. Retrieved 7/16 statements.
# Partially parsed test_prompt_and_delete_no_input_file_exists. Retrieved 9/18 statements.
# Partially parsed test_prompt_and_delete_with_input_yes_to_delete_dir. Retrieved 10/20 statements.
# Partially parsed test_prompt_and_delete_with_input_no_to_delete_reuse_true. Retrieved 8/17 statements.
# Partially parsed test_prompt_and_delete_with_input_no_to_delete_reuse_false_exits. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'os.path.isdir'
    var_2 = True
    var_3 = lambda p: var_2
    var_4 = 'cookiecutter.prompt.rmtree'
    var_5 = None
    var_6 = lambda p: var_5

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'content'
    var_2 = 'os.path.isdir'
    var_3 = False
    var_4 = lambda p: var_3
    var_5 = 'os.remove'
    var_6 = None
    var_7 = lambda p: var_6
    var_8 = True

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'os.path.isdir'
    var_2 = True
    var_3 = lambda p: var_2
    var_4 = 'cookiecutter.prompt.read_user_yes_no'
    var_5 = lambda q, d: var_2
    var_6 = 'cookiecutter.prompt.rmtree'
    var_7 = None
    var_8 = lambda p: var_7
    var_9 = False

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'os.path.isdir'
    var_2 = True
    var_3 = lambda p: var_2
    var_4 = 'cookiecutter.prompt.read_user_yes_no'
    var_5 = 'delete'
    var_6 = False
    var_7 = lambda q, d: var_6 if var_5 in q else var_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'os.path.isdir'
    var_2 = True
    var_3 = lambda p: var_2
    var_4 = 'cookiecutter.prompt.read_user_yes_no'
    var_5 = False
    var_6 = lambda q, d: var_5
    var_7 = 'sys.exit'
    var_8 = None
    var_9 = lambda x: var_8



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_prompt_and_delete_true_when_no_input_is_true. Retrieved 7/12 statements.


import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'test_dir_to_delete'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = True
    var_7 = module_1.prompt_and_delete(var_3, var_6)
    assert var_7 is True
    var_8 = var_3.rmdir()



# Parsed testcases at query #3
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
    var_0 = '{"key": "value", missing_quote}'
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_read_user_variable_returns_default_when_prompt_is_none. Retrieved 3/6 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt_from_dict. Retrieved 7/10 statements.
# Partially parsed test_read_user_variable_applies_prefix. Retrieved 5/8 statements.
# Partially parsed test_read_user_variable_ignores_empty_prompt_string_in_dict. Retrieved 6/9 statements.


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
    var_4 = 'default_val'
    var_5 = module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'user_input'
    var_6 = 'Custom Question'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'def'
    var_2 = 'PRE_: '
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'input'
    var_4 = 'PRE_: var'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'def'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'valid_input'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'def'
    var_5 = module_0.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'val'



# Parsed testcases at query #5
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
    var_1 = 'hello'
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == 'hello'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = module_1.render_variable(var_0, var_4, var_3)
    assert var_5 == 'my_project'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'Peanut Butter Cookie'
    var_3 = {var_1: var_2}
    var_4 = "{{ cookiecutter.project_name.replace(' ', '_') }}"
    var_5 = module_1.render_variable(var_0, var_4, var_3)
    assert var_5 == 'Peanut_Butter_Cookie'

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'val'
    var_2 = 'foo'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.val }}'
    var_5 = 'bar'
    var_6 = [var_4, var_5]
    var_7 = module_1.render_variable(var_0, var_6, var_3)
    var_8 = bool(var_7 == ['foo', 'bar'])
    assert var_8 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'key_{{ cookiecutter.name }}'
    var_5 = '{{ cookiecutter.name }}'
    var_6 = {var_4: var_5}
    var_7 = module_1.render_variable(var_0, var_6, var_3)
    var_8 = bool(var_7 == {'key_test': 'test'})
    assert var_8 is True

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 123
    var_2 = {}
    var_3 = module_1.render_variable(var_0, var_1, var_2)
    assert var_3 == '123'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_read_user_yes_no_uses_var_name_when_no_prompts. Retrieved 6/10 statements.
# Partially parsed test_read_user_yes_no_uses_prompt_from_dict. Retrieved 6/10 statements.
# Partially parsed test_read_user_yes_no_uses_var_name_when_prompt_key_missing. Retrieved 8/12 statements.
# Partially parsed test_read_user_yes_no_handles_empty_prompt_string. Retrieved 5/9 statements.
# Partially parsed test_read_user_yes_no_with_prefix_and_prompt. Retrieved 7/11 statements.


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
    var_0 = 'save'
    var_1 = 'Do you want to save changes?'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = ''
    var_5 = module_0.read_user_yes_no(var_0, var_3, var_2, var_4)
    assert var_5 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'other_key'
    var_1 = 'Different prompt'
    var_2 = {var_0: var_1}
    var_3 = 'missing_key'
    var_4 = False
    var_5 = 'Prefix: '
    var_6 = module_0.read_user_yes_no(var_3, var_4, var_2, var_5)
    assert var_6 is True
    var_7 = 'Prefix: missing_key'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'empty'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.read_user_yes_no(var_0, var_3, var_2, var_1)
    assert var_4 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'delete'
    var_1 = 'Delete file?'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'CONFIRM: '
    var_5 = module_0.read_user_yes_no(var_0, var_3, var_2, var_4)
    assert var_5 is False
    var_6 = 'CONFIRM: Delete file?'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_read_user_dict_valid_input. Retrieved 13/16 statements.
# Partially parsed test_read_user_dict_uses_var_name_as_question. Retrieved 6/9 statements.
# Partially parsed test_read_user_dict_no_prompts_provided. Retrieved 9/12 statements.
# Partially parsed test_read_user_dict_with_empty_prompt_mapping. Retrieved 8/11 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'test_var'
    var_3 = 'default'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'Custom Prompt'
    var_7 = {var_2: var_6}
    var_8 = 'Pref: '
    var_9 = module_0.read_user_dict(var_2, var_5, var_7, var_8)
    var_10 = bool(var_9 == {'key': 'value'})
    assert var_10 is True
    var_11 = 'Pref: Custom Prompt [cyan bold]([DEFAULT_DISPLAY])[/]'
    var_12 = {var_3: var_4}
    var_13 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'simple_var'
    var_1 = {}
    var_2 = module_0.read_user_dict(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = 'simple_var [cyan bold]([DEFAULT_DISPLAY])[/]'
    var_5 = {}
    var_6 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'var'
    var_3 = {}
    var_4 = None
    var_5 = module_0.read_user_dict(var_2, var_3, var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True
    var_7 = 'var [cyan bold]([DEFAULT_DISPLAY])[/]'
    var_8 = {}
    var_9 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'not_a_dict'
    var_2 = module_0.read_user_dict(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = {}
    var_2 = ''
    var_3 = {var_0: var_2}
    var_4 = module_0.read_user_dict(var_0, var_1, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True
    var_6 = 'var [cyan bold]([DEFAULT_DISPLAY])[/]'
    var_7 = {}
    var_8 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_prompt_choice_for_config_no_input_returns_first_option. Retrieved 10/17 statements.
# Partially parsed test_prompt_choice_for_config_no_input_raises_error_on_empty_options. Retrieved 4/8 statements.
# Partially parsed test_prompt_choice_for_config_calls_read_user_choice_when_input_is_required. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{ cookiecutter.project_name }}'
    var_4 = 'other_option'
    var_5 = [var_3, var_4]
    var_6 = 'some_key'
    var_7 = lambda cookiecutter: var_1
    var_8 = lambda cookiecutter: var_4
    var_9 = True

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'some_key'
    var_3 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '{{ cookiecutter.project_name }}'
    var_4 = 'option2'
    var_5 = [var_3, var_4]
    var_6 = 'some_key'
    var_7 = False
    var_8 = 'some_key'
    var_9 = 'Custom Prompt'
    var_10 = {var_8: var_9}
    var_11 = '[TEST] '
    var_12 = [var_1, var_4]
    var_13 = {var_8: var_9}



# Parsed testcases at query #11
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
    var_13 = '  NO  '
    var_14 = var_0.process_response(var_13)
    assert var_14 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)
    var_3 = ''
    var_4 = var_0.process_response(var_3)
    var_5 = '123'
    var_6 = var_0.process_response(var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_user_variable_returns_default_when_no_input. Retrieved 3/5 statements.
# Partially parsed test_read_user_variable_uses_custom_prompt. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_uses_prefix. Retrieved 5/7 statements.
# Partially parsed test_read_user_variable_retries_on_none. Retrieved 5/6 statements.
# Partially parsed test_read_user_variable_with_empty_prompts_dict. Retrieved 4/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'default'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'default'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Please enter your name'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = module_0.read_user_variable(var_0, var_3, var_2)
    assert var_4 == 'John'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'default'
    var_2 = 'Enter: '
    var_3 = module_0.read_user_variable(var_0, var_1, prefix=var_2)
    assert var_3 == 'John'
    var_4 = 'Enter: name'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'recovered'
    var_2 = 'name'
    var_3 = 'default'
    var_4 = module_0.read_user_variable(var_2, var_3)
    assert var_4 == 'recovered'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'default'
    var_2 = {}
    var_3 = module_0.read_user_variable(var_0, var_1, var_2)
    assert var_3 == 'val'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_read_user_choice_returns_first_option_on_default. Retrieved 6/8 statements.
# Partially parsed test_read_user_choice_returns_specific_option. Retrieved 6/7 statements.
# Partially parsed test_read_user_choice_with_string_prompt. Retrieved 7/9 statements.
# Partially parsed test_read_user_choice_with_dict_prompt_and_custom_labels. Retrieved 14/16 statements.
# Partially parsed test_read_user_choice_with_prefix. Retrieved 7/11 statements.


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
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = 'cherry'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'fruit'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'banana'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'banana'
    var_2 = [var_0, var_1]
    var_3 = 'fruit'
    var_4 = 'Which fruit do you like?'
    var_5 = {var_3: var_4}
    var_6 = module_0.read_user_choice(var_3, var_2, var_5)
    assert var_6 == 'apple'
    var_7 = 'Which fruit do you like?'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '1'
    var_1 = 'apple'
    var_2 = 'banana'
    var_3 = [var_1, var_2]
    var_4 = 'fruit'
    var_5 = '__prompt__'
    var_6 = '1'
    var_7 = '2'
    var_8 = 'Pick a flavor:'
    var_9 = 'Sweet Apple'
    var_10 = 'Sour Banana'
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = module_0.read_user_choice(var_4, var_3, var_12)
    assert var_13 == 'apple'
    var_14 = 'Pick a flavor:'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = [var_0]
    var_2 = 'fruit'
    var_3 = '[QUERY] '
    var_4 = module_0.read_user_choice(var_2, var_1, prefix=var_3)
    assert var_4 == 'apple'
    var_5 = 0
    var_6 = '[QUERY] Select fruit'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = 'opt1'
    var_2 = 'opt2'
    var_3 = [var_1, var_2]
    var_4 = 'test_var'
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.read_user_choice(var_0, var_3, var_8)
    assert var_9 == 'opt1'
    var_10 = '__prompt__'
    var_11 = bool('__prompt__' not in var_8[var_0])
    assert var_11 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_choose_nested_template_new_style_success. Retrieved 21/25 statements.
# Partially parsed test_choose_nested_template_old_style_success. Retrieved 16/20 statements.
# Partially parsed test_choose_nested_template_illegal_path_raises_error. Retrieved 12/15 statements.
# Partially parsed test_choose_nested_template_none_path_raises_error. Retrieved 12/15 statements.


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
    var_12 = '/tmp/repo'
    var_13 = '/tmp/repo'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Path(*var_14, **var_15)
    var_17 = '/tmp/repo/templates/option1'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.Path(*var_18, **var_19)
    var_21 = True
    var_22 = module_1.choose_nested_template(var_11, var_12, var_21)
    var_23 = [var_17]
    var_24 = {}
    var_25 = module_0.Path(*var_23, **var_24)
    var_26 = str(var_25)
    var_27 = bool(var_22 == var_26)
    assert var_27 is True

import pathlib as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice1 (templates/choice1)'
    var_3 = 'choice2 (templates/choice2)'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = '/tmp/repo'
    var_8 = '/tmp/repo'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)
    var_12 = '/tmp/repo/templates/choice1'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_0.Path(*var_13, **var_14)
    var_16 = True
    var_17 = module_1.choose_nested_template(var_6, var_7, var_16)
    var_18 = [var_12]
    var_19 = {}
    var_20 = module_0.Path(*var_18, **var_19)
    var_21 = str(var_20)
    var_22 = bool(var_17 == var_21)
    assert var_22 is True

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
    var_9 = '/tmp/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_prompt_for_config_calls_read_user_variable_when_not_no_input. Retrieved 7/9 statements.
# Partially parsed test_prompt_for_config_calls_read_user_choice_when_list_is_present. Retrieved 9/11 statements.
# Partially parsed test_prompt_for_config_calls_read_user_yes_no_for_bool. Retrieved 7/9 statements.
# Partially parsed test_prompt_for_config_calls_read_user_dict_when_dict_is_present. Retrieved 10/12 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_internal_var'
    var_3 = 'my_project'
    var_4 = 'secret'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'my_project'
    var_10 = var_8['_internal_var']
    assert var_10 == 'secret'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = '_internal'
    var_4 = 'My Project'
    var_5 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_6 = 'val'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.prompt_for_config(var_8, var_9)
    var_11 = var_10['project_name']
    assert var_11 == 'My Project'
    var_12 = var_10['repo_name']
    assert var_12 == 'My_Project'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'type'
    var_2 = '_internal'
    var_3 = 'web'
    var_4 = 'api'
    var_5 = 'cli'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'val'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.prompt_for_config(var_9, var_10)
    var_12 = var_11['type']
    assert var_12 == 'web'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'settings'
    var_2 = '_internal'
    var_3 = 'debug'
    var_4 = 'port'
    var_5 = 'true'
    var_6 = '8080'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'val'
    var_9 = {var_1: var_7, var_2: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.prompt_for_config(var_10, var_11)
    var_13 = var_12['settings']['debug']
    assert var_13 == 'true'
    var_14 = var_12['settings']['port']
    assert var_14 == '8080'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__template_var__'
    var_3 = 'test'
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['__template_var__']
    assert var_9 == 'test'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'user_input'
    var_2 = 'default_val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = var_6['user_input']
    assert var_7 == 'user_provided_val'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['choice_var']
    assert var_9 == 'option2'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = var_6['is_enabled']
    assert var_7 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'config_dict'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val'
    var_8 = False
    var_9 = module_0.prompt_for_config(var_6, var_8)
    var_10 = var_9['config_dict']
    var_11 = bool(var_9['config_dict'] == {'key': 'new_val'})
    assert var_11 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_prompt_for_config_no_input_skips_read_user_dict. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'my_dict'
    var_2 = '_private_var'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'hidden'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = 'key'
    var_11 = 'value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_prompt_for_config_no_input_simple_values. Retrieved 9/11 statements.
# Partially parsed test_prompt_for_config_no_input_with_templates. Retrieved 10/12 statements.
# Partially parsed test_prompt_for_config_raises_undefined_error. Retrieved 8/13 statements.
# Partially parsed test_prompt_for_config_handles_lists_as_choices. Retrieved 9/11 statements.
# Partially parsed test_prompt_for_config_with_prompts_extraction. Retrieved 10/12 statements.
# Partially parsed test_prompt_for_config_with_boolean_with_input. Retrieved 7/9 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_private_var'
    var_3 = 'my_project'
    var_4 = 'hidden'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['project_name']
    assert var_9 == 'my_project'
    var_10 = var_8['_private_var']
    assert var_10 == 'hidden'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'repo_name'
    var_3 = 'my_project'
    var_4 = '{{ cookiecutter.project_name.replace(" ", "_") }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'my_project'
    var_8 = True
    var_9 = module_0.prompt_for_config(var_6, var_8)
    var_10 = var_9['repo_name']
    assert var_10 == 'my_project'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '{{ non_existent_var }}'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'Undefined variable'
    var_6 = True
    var_7 = module_0.prompt_for_config(var_4, var_6)
    var_8 = "Unable to render variable 'project_name'"

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'version'
    var_2 = '1.0'
    var_3 = '2.0'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['version']
    assert var_9 == '1.0'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'config_dict'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_0.prompt_for_config(var_6, var_7)
    var_9 = var_8['config_dict']
    var_10 = bool(var_8['config_dict'] == {'key': 'value'})
    assert var_10 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '__prompts__'
    var_3 = 'my_project'
    var_4 = 'Enter project name: '
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)
    var_10 = var_9['project_name']
    assert var_10 == 'new_name'
    var_11 = '__prompts__'
    var_12 = bool('__prompts__' not in var_7['cookiecutter'])
    assert var_12 is True

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
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = var_6['is_enabled']
    assert var_7 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_prompt_and_delete_no_input_true. Retrieved 3/6 statements.
# Partially parsed test_prompt_and_delete_dir_deleted. Retrieved 2/5 statements.
# Partially parsed test_prompt_and_delete_file_deleted. Retrieved 2/5 statements.
# Partially parsed test_prompt_and_delete_do_not_delete_reuse_existing. Retrieved 4/7 statements.
# Partially parsed test_prompt_and_delete_do_not_delete_exit_program. Retrieved 3/6 statements.


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = True
    var_2 = module_0.prompt_and_delete(var_0, var_1)
    assert var_2 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.prompt_and_delete(var_0)
    assert var_1 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'test_file'
    var_1 = module_0.prompt_and_delete(var_0)
    assert var_1 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'test_dir'
    var_3 = module_0.prompt_and_delete(var_2)
    assert var_3 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = False
    var_1 = 'test_dir'
    var_2 = module_0.prompt_and_delete(var_1)
    assert var_2 is None



