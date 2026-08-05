####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test prompt_and_delete logic for deletion and reuse scenarios.'
    var_1 = '/tmp/fake_dir'
    var_2 = False
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'Test prompt_and_delete with no_input=True.'
    var_1 = '/tmp/fake_file'
    var_2 = True

def test_case_0():
    var_0 = 'Test that os.remove is called if the path is a file.'
    var_1 = '/tmp/fake_file'
    var_2 = False



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the prompt_for_config function by mocking user inputs \n    to verify the logic of variable rendering and dictionary construction.\n    '
    var_1 = 'read_user_variable'
    var_2 = 'read_user_yes_no'
    var_3 = 'read_user_dict'
    var_4 = 'prompt_choice_for_config'
    var_5 = 'create_env_with_context'
    var_6 = 'New Project'
    var_7 = True
    var_8 = 'key'
    var_9 = 'new_val'
    var_10 = {var_8: var_9}
    var_11 = 'opt1'
    var_12 = 'key'
    var_13 = 'new_val'
    var_14 = False

def test_case_0():
    var_0 = 'Tests prompt_for_config with no_input=True (automated mode).'
    var_1 = True



# Parsed testcases at query #3
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)
    var_3 = ''
    var_4 = var_0.process_response(var_3)



# Parsed testcases at query #4
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests for read_user_dict function.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'my_var'
    var_5 = module_0.read_user_dict(var_4, var_3)
    var_6 = 'my_var'
    var_7 = 'Enter your data'
    var_8 = {var_6: var_7}
    var_9 = 'PROMPT: '
    var_10 = 'a'
    var_11 = 1
    var_12 = {var_10: var_11}
    var_13 = 'my_var'
    var_14 = module_0.read_user_dict(var_13, var_12, var_8, var_9)
    var_15 = 'PROMPT:Enter your data [cyan bold](default)[/]'
    var_16 = 'my_var'
    var_17 = 'not a dict'
    var_18 = module_0.read_user_dict(var_16, var_17)
    var_19 = '{"invalid": json'
    var_20 = '[1, 2, 3]'
    var_21 = 'existing'
    var_22 = 'data'
    var_23 = {var_21: var_22}
    var_24 = 'my_var'
    var_25 = module_0.read_user_dict(var_24, var_23)



# Parsed testcases at query #5
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = 'bool'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.process_json(var_0)
    var_12 = '{"outer": {"inner": "data"}}'
    var_13 = 'outer'
    var_14 = 'inner'
    var_15 = 'data'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = module_0.process_json(var_12)
    var_19 = '{"key": "value",}'
    var_20 = module_0.process_json(var_19)
    var_21 = '["item1", "item2"]'
    var_22 = module_0.process_json(var_21)
    var_23 = '"just a string"'
    var_24 = module_0.process_json(var_23)
    var_25 = ''
    var_26 = module_0.process_json(var_25)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the process_response method of YesNoPrompt class.'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'Question?'
    var_2 = module_0.YesNoPrompt(var_1)
    var_3 = 'maybe'
    var_4 = var_2.process_response(var_3)
    var_5 = ''
    var_6 = var_2.process_response(var_5)



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Tests prompt_for_config with no_input=True to verify that it \n    renders variables and processes the context correctly without user interaction.\n    '
    var_1 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests that UndefinedError during rendering raises UndefinedVariableInTemplate.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '{{ cookiecutter.non_existent }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True
    var_7 = module_0.prompt_for_config(var_5, var_6)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests the interactive flow of prompt_for_config by mocking user inputs.\n    '
    var_1 = 'New Project'
    var_2 = '1.0.0'
    var_3 = '1'
    var_4 = 'y'
    var_5 = '{"new_key": "val"}'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'version'
    var_9 = 'options_list'
    var_10 = 'bool_var'
    var_11 = 'dict_var'
    var_12 = 'default_name'
    var_13 = '0.1.0'
    var_14 = 'opt1'
    var_15 = 'opt2'
    var_16 = [var_14, var_15]
    var_17 = True
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = {var_7: var_12, var_8: var_13, var_9: var_16, var_10: var_17, var_11: var_20}
    var_22 = {var_6: var_21}
    var_23 = False
    var_24 = module_0.prompt_for_config(var_22, var_23)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'my_project'
    var_1 = '0.1.0'
    var_2 = '1'
    var_3 = '{"author": "tester"}'
    var_4 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #10
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = '\n    Tests the prompt_for_config function with various scenarios including \n    simple variables, lists (choices), booleans, and no-input modes.\n    '
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests that prompt_for_config raises UndefinedVariableInTemplate \n    when a template variable cannot be rendered.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'broken_var'
    var_4 = '__prompts__'
    var_5 = 'my_project'
    var_6 = '{{ non_existent_variable }}'
    var_7 = {}
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'Undefined variable'
    var_11 = True
    var_12 = module_0.prompt_for_config(var_9, var_11)

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '\n    Tests the second pass of prompt_for_config which handles dictionary variables.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'metadata'
    var_3 = '__prompts__'
    var_4 = 'author'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = '__prompt__'
    var_8 = 'Enter metadata'
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_9}
    var_11 = {var_2: var_6, var_3: var_10}
    var_12 = {var_1: var_11}
    var_13 = module_0.Environment()
    var_14 = False
    var_15 = module_1.prompt_for_config(var_12, var_14)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = True
    var_1 = True
    var_2 = False
    var_3 = False
    var_4 = False



# Parsed testcases at query #12
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the process_response method of the YesNoPrompt class.'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'Question?'
    var_2 = module_0.YesNoPrompt(var_1)
    var_3 = 'maybe'
    var_4 = var_2.process_response(var_3)
    var_5 = ''
    var_6 = var_2.process_response(var_5)

def test_case_0():
    var_0 = 'Tests the .ask() static method via YesNoPrompt.'
    var_1 = 'Test?'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test prompt_and_delete handles deletion of files and directories based on user input.'
    var_1 = True
    var_2 = True
    var_3 = False
    var_4 = False
    var_5 = False



# Parsed testcases at query #14
#--------------------------


import cookiecutter.prompt as module_0
import collections as module_1

def test_case_0():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = 'bool'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.process_json(var_0)
    var_12 = '{"outer": {"inner": [1, 2, 3]}}'
    var_13 = 'outer'
    var_14 = 'inner'
    var_15 = 2
    var_16 = 3
    var_17 = [var_8, var_15, var_16]
    var_18 = (var_14, var_17)
    var_19 = [var_18]
    var_20 = module_0.process_json(var_12)
    var_21 = '{"key": "value",}'
    var_22 = module_0.process_json(var_21)
    var_23 = '"just a string"'
    var_24 = module_0.process_json(var_23)
    var_25 = '{}'
    var_26 = module_0.process_json(var_25)
    var_27 = module_1.OrderedDict()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the JsonPrompt class and its response processing.'
    var_1 = '{"key": "value", "number": 123, "list": [1, 2]}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'list'
    var_5 = 'value'
    var_6 = 123
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = {var_2: var_5, var_3: var_6, var_4: var_9}
    var_11 = '{}'
    var_12 = '{"key": "value",}'
    var_13 = '["item1", "item2"]'
    var_14 = 'Enter JSON'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the logic for deleting or reusing files based on user input.'
    var_1 = '/tmp/test_file'
    var_2 = False
    var_3 = module_0.prompt_and_delete(var_1, var_2)
    assert var_3 is True
    var_4 = 'no'
    var_5 = 'yes'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = "Specific test for the 'No to Delete, Yes to Reuse' flow."
    var_1 = '/tmp/test_file'
    var_2 = 'no'
    var_3 = 'yes'
    var_4 = False
    var_5 = module_0.prompt_and_delete(var_1, var_4)
    assert var_5 is False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = "Specific test for the 'No to Delete, No to Reuse' flow leading to sys.exit."
    var_1 = '/tmp/test_file'
    var_2 = 'no'
    var_3 = False
    var_4 = module_0.prompt_and_delete(var_1, var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the behavior when no_input is set to True.'
    var_1 = '/tmp/test_dir'
    var_2 = True
    var_3 = module_0.prompt_and_delete(var_1, var_2)
    assert var_3 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that os.remove is called when path is a file.'
    var_1 = '/tmp/test_file.txt'
    var_2 = True
    var_3 = module_0.prompt_and_delete(var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the process_response method of YesNoPrompt for various inputs.'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'Test Question'
    var_2 = module_0.YesNoPrompt(var_1)
    var_3 = 'maybe'
    var_4 = var_2.process_response(var_3)
    var_5 = ''
    var_6 = var_2.process_response(var_5)
    var_7 = 'unknown'
    var_8 = var_2.process_response(var_7)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the JsonPrompt class behavior, specifically its \n    static method process_response and its integration with process_json.\n    '
    var_1 = '{"key": "value", "number": 123}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'value'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '["item1", "item2"]'
    var_8 = '{"key": "value"'
    var_9 = 'a'
    var_10 = 1
    var_11 = '{}'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Question?'
    assert var_0 is True
    assert var_0 is False
    var_1 = module_0.YesNoPrompt(var_0)
    var_2 = 'maybe'
    var_3 = var_1.process_response(var_2)
    var_4 = ''
    var_5 = var_1.process_response(var_4)
    var_6 = 'random_string'
    var_7 = var_1.process_response(var_6)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\n    Tests prompt_and_delete with various user inputs and file scenarios.\n    Note: The function calls sys.exit() on certain logical paths, \n    so we must catch SystemExit.\n    '
    var_1 = 'test_dummy_file.txt'
    var_2 = 'test_dummy_dir'
    var_3 = 'content'
    var_4 = False
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Tests the behavior when no_input is set to True.'
    var_1 = 'test_dummy_file_no_input.txt'
    var_2 = 'content'
    var_3 = True



# Parsed testcases at query #21
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the process_response method of the YesNoPrompt class.'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'Test Question'
    var_2 = module_0.YesNoPrompt(var_1)
    var_3 = 'maybe'
    var_4 = var_2.process_response(var_3)
    var_5 = ''
    var_6 = var_2.process_response(var_5)
    var_7 = 'unknown_string'
    var_8 = var_2.process_response(var_7)



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = '/tmp/test_repo'
    var_2 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that an illegal template path raises ValueError.'
    var_1 = 'cookiecutter'
    var_2 = 'templates'
    var_3 = 'bad'
    var_4 = 'path'
    var_5 = '/absolute/path/is/illegal'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = '/tmp/repo'
    var_11 = True
    var_12 = module_0.choose_nested_template(var_9, var_10, var_11)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that error is raised if config keys are missing.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '/tmp/repo'
    var_5 = True
    var_6 = module_0.choose_nested_template(var_3, var_4, var_5)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the process_response method and JSON parsing logic of JsonPrompt.'
    var_1 = '{"key": "value", "number": 123, "bool": true}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'bool'
    var_5 = 'value'
    var_6 = 123
    var_7 = True
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = '{"outer": {"inner": [1, 2, 3]}}'
    var_10 = 'outer'
    var_11 = 'inner'
    var_12 = 2
    var_13 = 3
    var_14 = [var_7, var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {var_10: var_15}
    var_17 = '{"key": "value"'
    var_18 = '["item1", "item2"]'
    var_19 = ''



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests the prompt_for_config function by simulating both no_input=True \n    and no_input=False scenarios using patches for user interaction.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = '_private_var'
    var_5 = '__render_me__'
    var_6 = 'TestProject'
    var_7 = '0.1.0'
    var_8 = 'hidden'
    var_9 = 'Hello {{ cookiecutter.project_name }}'
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = {var_1: var_10}
    var_12 = True
    var_13 = module_0.prompt_for_config(var_11, var_12)
    var_14 = 'use_git'
    var_15 = 'options_list'
    var_16 = 'InteractiveProject'
    var_17 = True
    var_18 = 'opt1'
    var_19 = 'opt2'
    var_20 = [var_18, var_19]
    var_21 = {var_2: var_16, var_14: var_17, var_15: var_20}
    var_22 = {var_1: var_21}
    var_23 = 'InteractiveProject'
    var_24 = '1'
    var_25 = True
    var_26 = False
    var_27 = module_0.prompt_for_config(var_22, var_26)
    var_28 = 'broken_var'
    var_29 = 'ErrorProject'
    var_30 = '{{ cookiecutter.non_existent }}'
    var_31 = {var_25: var_29, var_28: var_30}
    var_32 = {var_24: var_31}
    var_33 = True
    var_34 = module_0.prompt_for_config(var_32, var_33)
    var_35 = str(var_33)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the process_response method of YesNoPrompt.'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'Question?'
    var_2 = module_0.YesNoPrompt(var_1)
    var_3 = 'maybe'
    var_4 = var_2.process_response(var_3)
    var_5 = ''
    var_6 = var_2.process_response(var_5)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123, "bool": true}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = 'bool'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.process_json(var_0)
    var_12 = '{"outer": {"inner": [1, 2, 3]}}'
    var_13 = 'outer'
    var_14 = 'inner'
    var_15 = 2
    var_16 = 3
    var_17 = [var_8, var_15, var_16]
    var_18 = (var_14, var_17)
    var_19 = [var_18]
    var_20 = '{"key": "value",}'
    var_21 = module_0.process_json(var_20)
    var_22 = '"just a string"'
    var_23 = module_0.process_json(var_22)
    var_24 = '[1, 2, 3]'
    var_25 = module_0.process_json(var_24)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Tests prompt_and_delete for various deletion scenarios.'
    var_1 = './test_temp_dir'
    var_2 = True
    var_3 = 'test_file.txt'
    var_4 = 'content'
    var_5 = 'test_subdir'
    var_6 = True
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = False



# Parsed testcases at query #5
#--------------------------


import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = '\n    Test prompt_choice_for_config with both no_input=True (returning first)\n    and no_input=False (calling read_user_choice).\n    '
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'My Project'
    var_4 = {var_2: var_3}
    var_5 = 'my_choice'
    var_6 = 'Option A'
    var_7 = '{{ cookiecutter.project_name }}'
    var_8 = 'Option C'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'my_choice'
    var_11 = 'Please select a preference'
    var_12 = {var_10: var_11}
    var_13 = '[test] '
    var_14 = True
    var_15 = module_1.prompt_choice_for_config(var_4, var_1, var_5, var_9, var_14)
    assert var_15 == 'Option A'
    var_16 = False
    var_17 = module_1.prompt_choice_for_config(var_4, var_1, var_5, var_9, var_16, var_12, var_13)
    assert var_17 == 'My Project'
    var_18 = 'Option A'
    var_19 = 'My Project'
    var_20 = 'Option C'
    var_21 = [var_18, var_19, var_20]

import jinja2.environment as module_0
import cookiecutter.prompt as module_1

def test_case_0():
    var_0 = 'Test that prompt_choice_for_config raises ValueError if options are empty and no_input is True.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'empty_key'
    var_4 = []
    var_5 = True
    var_6 = module_1.prompt_choice_for_config(var_2, var_1, var_3, var_4, var_5)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Question?'
    assert var_0 is True
    assert var_0 is False
    var_1 = module_0.YesNoPrompt(var_0)
    var_2 = 'maybe'
    var_3 = var_1.process_response(var_2)
    var_4 = ''
    var_5 = var_1.process_response(var_4)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_yes_no function for different user inputs.'
    var_1 = 'test_var'
    var_2 = False
    var_3 = module_0.read_user_yes_no(var_1, var_2)
    assert var_3 is True
    var_4 = 'test_var'
    var_5 = True
    var_6 = module_0.read_user_yes_no(var_4, var_5)
    assert var_6 is False
    var_7 = 'test_var'
    var_8 = False
    var_9 = module_0.read_user_yes_no(var_7, var_8)
    assert var_9 is True
    var_10 = 'test_var'
    var_11 = True
    var_12 = module_0.read_user_yes_no(var_10, var_11)
    assert var_12 is False
    var_13 = 'my_var'
    var_14 = 'Custom Question?'
    var_15 = {var_13: var_14}
    var_16 = 'my_var'
    var_17 = False
    var_18 = module_0.read_user_yes_no(var_16, var_17, var_15)
    assert var_18 is True
    var_19 = 'var'
    var_20 = False
    var_21 = 'PRE_'
    var_22 = module_0.read_user_yes_no(var_19, var_20, prefix=var_21)
    assert var_22 is True
    var_23 = 'var'
    var_24 = True
    var_25 = module_0.read_user_yes_no(var_23, var_24)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '{"key": "value", "number": 123, "bool": true, "list": [1, 2]}'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = (var_1, var_2)
    var_4 = 'number'
    var_5 = 123
    var_6 = (var_4, var_5)
    var_7 = 'bool'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = 'list'
    var_11 = 2
    var_12 = [var_8, var_11]
    var_13 = (var_10, var_12)
    var_14 = [var_3, var_6, var_9, var_13]
    var_15 = module_0.process_json(var_0)
    var_16 = '{"outer": {"inner": "val"}}'
    var_17 = 'outer'
    var_18 = 'inner'
    var_19 = 'val'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = '{"key": "value"'
    var_23 = module_0.process_json(var_22)
    var_24 = '[1, 2, 3]'
    var_25 = module_0.process_json(var_24)
    var_26 = '"hello"'
    var_27 = module_0.process_json(var_26)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '/tmp/repo'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 't1'
    var_3 = 'path'
    var_4 = '/absolute/path'
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
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/repo'
    var_6 = True
    var_7 = module_0.choose_nested_template(var_4, var_5, var_6)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests the prompt_for_config function by mocking the underlying \n    interactive prompts to simulate a non-interactive (no_input=True) execution.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_private'
    var_4 = 'use_git'
    var_5 = 'list_var'
    var_6 = '__prompts__'
    var_7 = 'my_project'
    var_8 = 'secret'
    var_9 = True
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = 'Project Name'
    var_14 = {var_2: var_13}
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_12, var_6: var_14}
    var_16 = {var_1: var_15}
    var_17 = True
    var_18 = module_0.prompt_for_config(var_16, var_17)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests if render_variable logic is correctly applied during config prompting.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'repo_name'
    var_4 = '__prompts__'
    var_5 = 'TestProject'
    var_6 = '{{ cookiecutter.project_name.lower() }}'
    var_7 = {}
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'TestProject'
    var_11 = False
    var_12 = module_0.prompt_for_config(var_9, var_11)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests that UndefinedError in rendering raises UndefinedVariableInTemplate.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '{{ non_existent }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'Undefined variable'
    var_7 = True
    var_8 = module_0.prompt_for_config(var_5, var_7)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'tpl1'
    var_3 = 'path'
    var_4 = '/absolute/path'
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
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/repo'
    var_4 = True
    var_5 = module_0.choose_nested_template(var_2, var_3, var_4)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Tests the JsonPrompt class functionality via its static process_response method.'
    var_1 = '{"key": "value", "number": 123}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'value'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '["item1", "item2"]'
    var_8 = '{"key": "value",}'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 't1'
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
    var_2 = 't1'
    var_3 = 't2'
    var_4 = 'path'
    var_5 = 'templates/a'
    var_6 = {var_4: var_5}
    var_7 = 'templates/b'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '/tmp/repo'
    var_13 = False
    var_14 = module_0.choose_nested_template(var_11, var_12, var_13)
    var_15 = '/tmp/repo/templates/b'
    var_16 = str(var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/repo'
    var_6 = True
    var_7 = module_0.choose_nested_template(var_4, var_5, var_6)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = module_0.YesNoPrompt()
    var_1 = 'maybe'
    var_2 = var_0.process_response(var_1)
    var_3 = ''
    var_4 = var_0.process_response(var_3)
    var_5 = 'random_string'
    var_6 = var_0.process_response(var_5)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Question?'
    assert var_0 is True
    assert var_0 is False
    var_1 = module_0.YesNoPrompt(var_0)
    var_2 = '  YES  '
    var_3 = var_1.process_response(var_2)
    assert var_3 is True
    var_4 = 'No\n'
    var_5 = var_1.process_response(var_4)
    assert var_5 is False
    var_6 = 'maybe'
    var_7 = var_1.process_response(var_6)
    var_8 = ''
    var_9 = var_1.process_response(var_8)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Tests prompt_and_delete for various user decision flows.'
    var_1 = 'test_file.txt'
    var_2 = 'dummy content'
    var_3 = False
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'Tests prompt_and_delete when no_input is True (automation mode).'
    var_1 = 'test_dir'
    var_2 = 'test_dir_content'
    var_3 = True
    var_4 = True

def test_case_0():
    var_0 = 'Tests that sys.exit is called when user refuses to delete and refuses to reuse.'
    var_1 = 'test_file.txt'
    var_2 = ''
    var_3 = False



# Parsed testcases at query #17
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'tpl1'
    var_3 = 'path'
    var_4 = '/absolute/path'
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
    var_2 = 'tpl1'
    var_3 = 'path'
    var_4 = 'tpl1_path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)
    var_12 = '/tmp/repo/tpl1_path'
    var_13 = str(var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'InvalidStringNoParens'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = False
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_value'
    var_3 = 'project_name'
    var_4 = 'Enter your project name:'
    var_5 = {var_3: var_4}
    var_6 = 'project_name'
    var_7 = 'default_val'
    var_8 = module_0.read_user_variable(var_6, var_7, var_5)
    assert var_8 == 'new_project'
    var_9 = 'Enter your project name:'
    var_10 = 'project_name'
    var_11 = 'default_val'
    var_12 = '[bold] '
    var_13 = module_0.read_user_variable(var_10, var_11, prefix=var_12)
    assert var_13 == 'prefixed_val'
    var_14 = '[bold] project_name'
    var_15 = 'project_name'
    var_16 = 'default_val'
    var_17 = module_0.read_user_variable(var_15, var_16)
    assert var_17 == 'valid_input'
    var_18 = 'app_id'
    var_19 = 'Application Identifier'
    var_20 = {var_18: var_19}
    var_21 = 'app_id'
    var_22 = 999
    var_23 = 'ID: '
    var_24 = module_0.read_user_variable(var_21, var_22, var_20, var_23)
    assert var_24 == '123'
    var_25 = 'ID: Application Identifier'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test prompt_and_delete logic for both file and directory deletion.'
    var_1 = 'test_file.txt'
    var_2 = 'test_dir'
    var_3 = 'content'
    var_4 = False

def test_case_0():
    var_0 = 'Test the flow where user refuses to delete but wants to reuse (sys.exit).'
    var_1 = 'stay_alive.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = False

def test_case_0():
    var_0 = 'Test prompt_and_delete with no_input=True (should delete regardless).'
    var_1 = 'auto_delete_dir'
    var_2 = True

def test_case_0():
    var_0 = 'Test the flow where user refuses to delete but wants to reuse (returns False).'
    var_1 = 'reuse_me.txt'
    var_2 = 'content'
    var_3 = False
    var_4 = True



# Parsed testcases at query #20
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests various scenarios for reading user variables.'
    var_1 = 'project_name'
    var_2 = 'default_val'
    var_3 = module_0.read_user_variable(var_1, var_2)
    assert var_3 == 'my_project'
    var_4 = 'project_name'
    var_5 = 'Enter your project title'
    var_6 = {var_4: var_5}
    var_7 = 'PROMPT: '
    var_8 = 'project_name'
    var_9 = 'default'
    var_10 = module_0.read_user_variable(var_8, var_9, var_6, var_7)
    assert var_10 == 'Awesome App'
    var_11 = 'PROMPT: Enter your project title'
    var_12 = None
    var_13 = 'recovered_value'
    var_14 = 'test_var'
    var_15 = 'default'
    var_16 = module_0.read_user_variable(var_14, var_15)
    assert var_16 == 'recovered_value'
    var_17 = 'unmapped_var'
    var_18 = 'default'
    var_19 = 'other_var'
    var_20 = 'Hello'
    var_21 = {var_19: var_20}
    var_22 = module_0.read_user_variable(var_17, var_18, var_21)
    assert var_22 == 'value'
    var_23 = 'var'
    var_24 = 'default'
    var_25 = None
    var_26 = module_0.read_user_variable(var_23, var_24, var_25)
    assert var_26 == 'value'
    var_27 = 'api_key'
    var_28 = 'Secret Key'
    var_29 = {var_27: var_28}
    var_30 = '[SECURE] '
    var_31 = 'api_key'
    var_32 = 'none'
    var_33 = module_0.read_user_variable(var_31, var_32, var_29, var_30)
    assert var_33 == '12345'
    var_34 = '[SECURE] Secret Key'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests read_user_yes_no for various input scenarios.'
    var_1 = 'test_var'
    var_2 = False
    var_3 = module_0.read_user_yes_no(var_1, var_2)
    assert var_3 is True
    var_4 = 'test_var'
    var_5 = True
    var_6 = module_0.read_user_yes_no(var_4, var_5)
    assert var_6 is False
    var_7 = 'test_var'
    var_8 = False
    var_9 = module_0.read_user_yes_no(var_7, var_8)
    assert var_9 is True
    var_10 = 'test_var'
    var_11 = True
    var_12 = module_0.read_user_yes_no(var_10, var_11)
    assert var_12 is False
    var_13 = 'my_key'
    var_14 = 'Custom Question'
    var_15 = {var_13: var_14}
    var_16 = 'my_key'
    var_17 = False
    var_18 = module_0.read_user_yes_no(var_16, var_17, var_15)
    assert var_18 is True
    var_19 = 'test_var'
    var_20 = True
    var_21 = 'PROMPT: '
    var_22 = module_0.read_user_yes_no(var_19, var_20, prefix=var_21)
    assert var_22 is False
    var_23 = 0
    var_24 = 'PROMPT: test_var'
    var_25 = 'simple_var'
    var_26 = False
    var_27 = {}
    var_28 = module_0.read_user_yes_no(var_25, var_26, var_27)
    assert var_28 is True



# Parsed testcases at query #22
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Specifically tests that the function loops until a non-None value is provided.'
    var_1 = None
    var_2 = 'final'
    var_3 = 'test'
    var_4 = 'default'
    var_5 = module_0.read_user_variable(var_3, var_4)
    assert var_5 == 'final'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = 'default'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_value'



