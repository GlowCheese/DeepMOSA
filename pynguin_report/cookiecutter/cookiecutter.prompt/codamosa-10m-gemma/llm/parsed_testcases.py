####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 't1'
    var_3 = 'path'
    var_4 = 'title'
    var_5 = '/abs/path'
    var_6 = 'T1'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = '/tmp/repo'
    var_12 = True
    var_13 = module_0.choose_nested_template(var_10, var_11, var_12)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/tmp/repo'
    var_4 = True
    var_5 = module_0.choose_nested_template(var_2, var_3, var_4)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 't1'
    var_3 = 't2'
    var_4 = 'path'
    var_5 = 'title'
    var_6 = 'tpl1'
    var_7 = 'T1'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'tpl2'
    var_10 = 'T2'
    var_11 = {var_4: var_9, var_5: var_10}
    var_12 = {var_2: var_8, var_3: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_0: var_13}
    var_15 = '/tmp/repo'
    var_16 = False
    var_17 = module_0.choose_nested_template(var_14, var_15, var_16)
    var_18 = '/tmp/repo/tpl2'
    var_19 = str(var_4)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'Option (extracted/path)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = False
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)
    var_9 = '/tmp/repo/extracted/path'
    var_10 = str(var_4)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'default'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = module_0.read_user_dict(var_0, var_3)
    var_5 = 'my_var'
    var_6 = 'Custom Question'
    var_7 = {var_5: var_6}
    var_8 = 'my_var'
    var_9 = 'a'
    var_10 = 1
    var_11 = {var_9: var_10}
    var_12 = module_0.read_user_dict(var_8, var_11, var_7)
    var_13 = '[bold]Prefix: [/]'
    var_14 = 'my_var'
    var_15 = {}
    var_16 = module_0.read_user_dict(var_14, var_15, prefix=var_13)
    var_17 = 0
    var_18 = 'my_var'
    var_19 = 'not'
    var_20 = 'a'
    var_21 = 'dict'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.read_user_dict(var_18, var_22)
    var_24 = 'my_var'
    var_25 = {}
    var_26 = module_0.read_user_dict(var_24, var_25)
    var_27 = 'my_var'
    var_28 = {}
    var_29 = module_0.read_user_dict(var_27, var_28)
    var_30 = 'my_var'
    var_31 = {}
    var_32 = module_0.read_user_dict(var_30, var_31)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'a'
    var_3 = 'path'
    var_4 = '/absolute/path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/tmp'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'invalid_format'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'opt1'
    var_3 = 'opt2'
    var_4 = 'path'
    var_5 = 't1'
    var_6 = {var_4: var_5}
    var_7 = 't2'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = '/tmp'
    var_13 = False
    var_14 = module_0.choose_nested_template(var_11, var_12, var_13)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'use_git'
    var_3 = '__prompts__'
    var_4 = 'my_project'
    var_5 = True
    var_6 = {}
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.prompt_for_config(var_8, var_5)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Template error'
    var_1 = 'cookiecutter'
    var_2 = 'broken_var'
    var_3 = '{{ non_existent_var }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = False
    var_7 = module_0.prompt_for_config(var_5, var_6)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 'cookiecutter'
    var_3 = 'my_dict'
    var_4 = '__prompts__'
    var_5 = 'subkey'
    var_6 = 'subval'
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = False
    var_12 = module_0.prompt_for_config(var_10, var_11)



# Parsed testcases at query #5
#--------------------------




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
    var_6 = 'random_string'
    var_7 = var_1.process_response(var_6)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Option A'
    var_1 = 'Option B'
    var_2 = 'Option C'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'my_var'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'Option B'
    var_6 = 'my_var'
    var_7 = 'Custom Question'
    var_8 = {var_6: var_7}



# Parsed testcases at query #8
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
    var_12 = '["item1", "item2"]'
    var_13 = module_0.process_json(var_12)
    var_14 = '{"key": "value", broken}'
    var_15 = module_0.process_json(var_14)
    var_16 = '"just a string"'
    var_17 = module_0.process_json(var_16)
    var_18 = '{}'
    var_19 = module_0.process_json(var_18)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_yes_no function behavior with various inputs.'
    var_1 = 'test_var'
    var_2 = False
    var_3 = module_0.read_user_yes_no(var_1, var_2)
    assert var_3 is True
    var_4 = 'test_var'
    var_5 = True
    var_6 = module_0.read_user_yes_no(var_4, var_5)
    assert var_6 is False
    var_7 = 'my_var'
    var_8 = 'Custom Question'
    var_9 = {var_7: var_8}
    var_10 = 'my_var'
    var_11 = False
    var_12 = module_0.read_user_yes_no(var_10, var_11, var_9)
    assert var_12 is True
    var_13 = 'test_var'
    var_14 = False
    var_15 = module_0.read_user_yes_no(var_13, var_14)
    assert var_15 is True
    var_16 = 'test_var'
    var_17 = False
    var_18 = module_0.read_user_yes_no(var_16, var_17)
    assert var_18 is True
    var_19 = 'test_var'
    var_20 = True
    var_21 = module_0.read_user_yes_no(var_19, var_20)
    assert var_21 is False
    var_22 = 'PROMPT: '
    var_23 = 'var'
    var_24 = False
    var_25 = module_0.read_user_yes_no(var_23, var_24, prefix=var_22)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test Question'
    assert var_0 is True
    assert var_0 is False
    var_1 = module_0.YesNoPrompt(var_0)
    var_2 = '  YES  '
    var_3 = var_1.process_response(var_2)
    assert var_3 is True
    var_4 = 'No'
    var_5 = var_1.process_response(var_4)
    assert var_5 is False
    var_6 = 'maybe'
    var_7 = var_1.process_response(var_6)



# Parsed testcases at query #11
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
    var_12 = '{"outer": {"inner": "content"}}'
    var_13 = 'outer'
    var_14 = 'inner'
    var_15 = 'content'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = module_0.process_json(var_12)
    var_19 = '{"key": "value",}'
    var_20 = module_0.process_json(var_19)
    var_21 = '["item1", "item2"]'
    var_22 = module_0.process_json(var_21)
    var_23 = '"just a string"'
    var_24 = module_0.process_json(var_23)
    var_25 = '123'
    var_26 = module_0.process_json(var_25)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that read_user_yes_no correctly parses various truthy and falsy inputs.'
    var_1 = 'test_var'
    var_2 = True
    var_3 = module_0.read_user_yes_no(var_1, var_2)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that read_user_yes_no uses the custom prompt from the prompts dictionary.'
    var_1 = 'my_var'
    var_2 = 'Custom Question'
    var_3 = {var_1: var_2}
    var_4 = 'my_var'
    var_5 = True
    var_6 = module_0.read_user_yes_no(var_4, var_5, var_3)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that read_user_yes_no uses the var_name itself if no prompts are provided.'
    var_1 = 'simple_var'
    var_2 = False
    var_3 = module_0.read_user_yes_no(var_1, var_2)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that the prefix is correctly prepended to the question.'
    var_1 = 'var'
    var_2 = True
    var_3 = 'PRE: '
    var_4 = module_0.read_user_yes_no(var_1, var_2, prefix=var_3)
    var_5 = 0
    var_6 = 'PRE: var'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests prompt_for_config with no_input=True to verify the logic of \n    rendering variables and populating the cookiecutter dictionary\n    without interactive prompts.\n    '
    var_1 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests prompt_for_config when a list (choice) is present.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'template_type'
    var_3 = 'web'
    var_4 = 'api'
    var_5 = 'cli'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = False
    var_10 = module_0.prompt_for_config(var_8, var_9)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests that UndefinedError is correctly caught and re-raised \n    as UndefinedVariableInTemplate.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'broken_var'
    var_3 = '{{ non_existent_variable }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = True
    var_7 = module_0.prompt_for_config(var_5, var_6)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests the second pass of prompt_for_config which handles dictionary types.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'settings'
    var_3 = 'mode'
    var_4 = 'debug'
    var_5 = 'production'
    var_6 = False
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = False
    var_11 = module_0.prompt_for_config(var_9, var_10)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the prompt_and_delete function for various scenarios.'
    var_1 = True
    var_2 = True
    var_3 = False
    var_4 = module_0.prompt_and_delete(var_2, var_3)
    assert var_4 is True
    var_5 = False
    var_6 = module_0.prompt_and_delete(var_2, var_5)
    assert var_6 is False
    var_7 = False
    var_8 = module_0.prompt_and_delete(var_2, var_7)
    assert var_8 is None



# Parsed testcases at query #15
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the deletion logic of prompt_and_delete.'
    var_1 = '/tmp/test_dir'
    var_2 = False
    var_3 = module_0.prompt_and_delete(var_0, var_2)
    assert var_3 is True
    var_4 = False
    var_5 = module_0.prompt_and_delete(var_0, var_4)
    assert var_5 is False
    var_6 = False
    var_7 = module_0.prompt_and_delete(var_0, var_6)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that prompt_and_delete deletes automatically when no_input is True.'
    var_1 = '/tmp/test_file.txt'
    var_2 = True
    var_3 = module_0.prompt_and_delete(var_0, var_2)
    assert var_3 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test that the function handles files correctly using os.remove.'
    var_1 = '/tmp/test_file.txt'
    var_2 = False
    var_3 = module_0.prompt_and_delete(var_0, var_2)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Option A'
    var_1 = 'Option B'
    var_2 = 'Option C'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'my_var'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'Option B'
    var_6 = 'my_var'
    var_7 = 'Please choose a flavor'
    var_8 = {var_6: var_7}
    var_9 = 'my_var'
    var_10 = module_0.read_user_choice(var_9, var_3, var_8)
    assert var_10 == 'Option A'
    var_11 = '__prompt__'
    var_12 = '1'
    var_13 = 'Pick one'
    var_14 = 'First'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {var_6: var_15}
    var_17 = 'my_var'
    var_18 = module_0.read_user_choice(var_17, var_3, var_16)
    assert var_18 == 'Option A'
    var_19 = 0
    var_20 = 'my_var'
    var_21 = '[blue]PROMPT:[/] '
    var_22 = module_0.read_user_choice(var_20, var_3, prefix=var_21)
    assert var_22 == 'Option C'
    var_23 = 'my_var'
    var_24 = []
    var_25 = module_0.read_user_choice(var_23, var_24)
    var_26 = 'my_var'
    var_27 = module_0.read_user_choice(var_26, var_3)
    assert var_27 == 'Option A'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Tests prompt_for_config by mocking the interactive Prompt.ask calls.\n    We simulate a non-interactive flow (no_input=True) to verify \n    the logic of variable rendering and dictionary construction.\n    '
    var_1 = True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests prompt_for_config with simulated user input via mocking Prompt.ask.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'user_input'
    var_3 = '__prompts__'
    var_4 = 'default_val'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests that UndefinedError in rendering raises UndefinedVariableInTemplate.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'broken_var'
    var_3 = '__prompts__'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = False
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests behavior when a choice variable is empty.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'choices'
    var_3 = '__prompts__'
    var_4 = []
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)



# Parsed testcases at query #19
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
    var_20 = var_12
    var_21 = module_0.process_json(var_20)
    var_22 = '{"key": "value",}'
    var_23 = module_0.process_json(var_22)
    var_24 = '["item1", "item2"]'
    var_25 = module_0.process_json(var_24)
    var_26 = '"just a string"'
    var_27 = module_0.process_json(var_26)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = None
    var_1 = 'my_value'
    var_2 = 'var_name'
    var_3 = 'default'
    var_4 = module_0.read_user_variable(var_2, var_3)
    assert var_4 == 'my_value'
    var_5 = 'project_name'
    var_6 = 'Enter your project name:'
    var_7 = {var_5: var_6}
    var_8 = 'project_name'
    var_9 = 'default'
    var_10 = module_0.read_user_variable(var_8, var_9, var_7)
    assert var_10 == 'Project X'
    var_11 = 'Enter your project name:'
    var_12 = 'var_name'
    var_13 = 'default'
    var_14 = '[bold]Prompt: '
    var_15 = module_0.read_user_variable(var_12, var_13, prefix=var_14)
    assert var_15 == 'Value'
    var_16 = '[bold]Prompt: var_name'
    var_17 = 'simple_var'
    var_18 = 'default'
    var_19 = None
    var_20 = module_0.read_user_variable(var_17, var_18, var_19)
    assert var_20 == 'Simple'
    var_21 = 'var'
    var_22 = 'default'
    var_23 = module_0.read_user_variable(var_21, var_22)
    assert var_23 == 'Immediate'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_variable function for various input scenarios.'
    var_1 = 'project_name'
    var_2 = 'default_val'
    var_3 = module_0.read_user_variable(var_1, var_2)
    assert var_3 == 'my_project'
    var_4 = 'project_name'
    var_5 = 'Enter your project title:'
    var_6 = {var_4: var_5}
    var_7 = 'project_name'
    var_8 = 'default'
    var_9 = module_0.read_user_variable(var_7, var_8, var_6)
    assert var_9 == 'New Title'
    var_10 = 'Enter your project title:'
    var_11 = 'var'
    var_12 = 'def'
    var_13 = 'PRE_: '
    var_14 = module_0.read_user_variable(var_11, var_12, prefix=var_13)
    assert var_14 == 'value'
    var_15 = 'PRE_: var'
    var_16 = None
    var_17 = 'valid'
    var_18 = 'var'
    var_19 = 'def'
    var_20 = module_0.read_user_variable(var_18, var_19)
    assert var_20 == 'valid'
    var_21 = 'empty_prompt'
    var_22 = ''
    var_23 = {var_21: var_22}
    var_24 = 'empty_prompt'
    var_25 = 'def'
    var_26 = module_0.read_user_variable(var_24, var_25, var_23)
    assert var_26 == 'test'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test the prompt_and_delete function for various scenarios.'
    var_1 = True
    var_2 = True
    var_3 = False
    var_4 = False
    var_5 = False



# Parsed testcases at query #23
#--------------------------




####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '/fake/path/to/file'
    var_1 = False
    var_2 = True
    var_3 = False

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the branch where path is a file, not a directory.'
    var_1 = '/fake/file.txt'
    var_2 = False
    var_3 = module_0.prompt_and_delete(var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the process_response method of YesNoPrompt for various inputs.'
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



# Parsed testcases at query #4
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = '\n    Tests the prompt_for_config function by mocking user input \n    and verifying that variables are processed and rendered correctly.\n    '
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'use_git'
    var_4 = 'options'
    var_5 = 'metadata'
    var_6 = '__prompts__'
    var_7 = 'my_project'
    var_8 = True
    var_9 = 'opt1'
    var_10 = 'opt2'
    var_11 = [var_9, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 'Enter Project Name'
    var_16 = {var_2: var_15}
    var_17 = {var_2: var_7, var_3: var_8, var_4: var_11, var_5: var_14, var_6: var_16}
    var_18 = {var_1: var_17}
    var_19 = 'new_name'
    var_20 = True
    var_21 = '1'
    var_22 = '{"key": "new_value"}'
    var_23 = False
    var_24 = module_0.prompt_for_config(var_18, var_23)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests prompt_for_config with no_input=True (automation mode).'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'my_project'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests that UndefinedError in template rendering raises UndefinedVariableInTemplate.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '{{ non_existent_var }}'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = False
    var_7 = module_0.prompt_for_config(var_5, var_6)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = "Tests the behavior of JsonPrompt's process_response method."
    var_1 = '{"key": "value", "number": 123}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'value'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '["item1", "item2"]'
    var_8 = '{"key": "value"'
    var_9 = ''



# Parsed testcases at query #6
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Option A'
    var_1 = 'Option B'
    var_2 = 'Option C'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'my_var'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'Option A'
    var_6 = 'my_var'
    var_7 = 'Please pick one:'
    var_8 = {var_6: var_7}
    var_9 = 'my_var'
    var_10 = module_0.read_user_choice(var_9, var_3, var_8)
    assert var_10 == 'Option B'
    var_11 = '__prompt__'
    var_12 = '1'
    var_13 = '2'
    var_14 = '3'
    var_15 = 'Custom Question'
    var_16 = 'Friendly A'
    var_17 = 'Friendly B'
    var_18 = 'Friendly C'
    var_19 = {var_11: var_15, var_12: var_16, var_13: var_17, var_14: var_18}
    var_20 = {var_6: var_19}
    var_21 = 'my_var'
    var_22 = module_0.read_user_choice(var_21, var_3, var_20)
    assert var_22 == 'Option C'
    var_23 = 0
    var_24 = 'my_var'
    var_25 = '[RED] '
    var_26 = module_0.read_user_choice(var_24, var_3, prefix=var_25)
    assert var_26 == 'Option B'
    var_27 = 'my_var'
    var_28 = []
    var_29 = module_0.read_user_choice(var_27, var_28)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Option A'
    var_1 = 'Option B'
    var_2 = 'Option C'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'my_var'
    var_5 = module_0.read_user_choice(var_4, var_3)
    assert var_5 == 'Option A'
    var_6 = 'my_var'
    var_7 = 'Please pick a fruit'
    var_8 = {var_6: var_7}
    var_9 = 'my_var'
    var_10 = module_0.read_user_choice(var_9, var_3, var_8)
    assert var_10 == 'Option B'
    var_11 = '__prompt__'
    var_12 = '1'
    var_13 = '2'
    var_14 = '3'
    var_15 = 'Choose your destiny'
    var_16 = 'First Choice'
    var_17 = 'Second Choice'
    var_18 = 'Third Choice'
    var_19 = {var_11: var_15, var_12: var_16, var_13: var_17, var_14: var_18}
    var_20 = {var_6: var_19}
    var_21 = 'my_var'
    var_22 = module_0.read_user_choice(var_21, var_3, var_20)
    assert var_22 == 'Option C'
    var_23 = 0
    var_24 = 'empty'
    var_25 = []
    var_26 = module_0.read_user_choice(var_24, var_25)
    var_27 = 'var'
    var_28 = '[bold]PROMPT: [/]'
    var_29 = module_0.read_user_choice(var_27, var_3, prefix=var_28)
    assert var_29 == 'Option A'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = 'template'
    var_2 = 'cookiecutter'

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
    var_1 = 'template'
    var_2 = 'invalid_string'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/tmp/repo'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)

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
    var_13 = str(var_4)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_project'
    var_3 = 'project_name'
    var_4 = 'default_val'
    var_5 = module_0.read_user_variable(var_3, var_4)
    assert var_5 == 'valid_input'
    var_6 = 'project_name'
    var_7 = 'Enter your project name:'
    var_8 = {var_6: var_7}
    var_9 = 'project_name'
    var_10 = 'default'
    var_11 = module_0.read_user_variable(var_9, var_10, var_8)
    assert var_11 == 'decorated'
    var_12 = 'Enter your project name:'
    var_13 = 'simple_var'
    var_14 = 'default'
    var_15 = module_0.read_user_variable(var_13, var_14)
    assert var_15 == 'simple'
    var_16 = 'var'
    var_17 = 'def'
    var_18 = '[bold]Prefix: '
    var_19 = module_0.read_user_variable(var_16, var_17, prefix=var_18)
    assert var_19 == 'prefixed'
    var_20 = '[bold]Prefix: var'
    var_21 = 'var'
    var_22 = 'def'
    var_23 = {}
    var_24 = module_0.read_user_variable(var_21, var_22, var_23)
    assert var_24 == 'empty_dict_test'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_value'
    var_3 = 'var_name'
    var_4 = 'Custom Question'
    var_5 = {var_3: var_4}
    var_6 = 'var_name'
    var_7 = 'default_val'
    var_8 = module_0.read_user_variable(var_6, var_7, var_5)
    assert var_8 == 'user_input'
    var_9 = 'Custom Question'
    var_10 = {var_6: var_7}
    var_11 = 'Prefix: '
    var_12 = 'var_name'
    var_13 = 'default_val'
    var_14 = module_0.read_user_variable(var_12, var_13, var_10, var_11)
    assert var_14 == 'user_input'
    var_15 = 'Prefix: Custom Question'
    var_16 = 'var_name'
    var_17 = 'default_val'
    var_18 = module_0.read_user_variable(var_16, var_17)
    assert var_18 == 'recovered_value'
    var_19 = 'other_var'
    var_20 = 'Other Question'
    var_21 = {var_19: var_20}
    var_22 = 'var_name'
    var_23 = 'default_val'
    var_24 = module_0.read_user_variable(var_22, var_23, var_21)
    assert var_24 == 'val'
    var_25 = 'var_name'
    var_26 = 'default_val'
    var_27 = None
    var_28 = module_0.read_user_variable(var_25, var_26, var_27)
    assert var_28 == 'val'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_val'
    var_3 = 'var_name'
    var_4 = 'Custom Question'
    var_5 = {var_3: var_4}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Tests the process_response static method of JsonPrompt.'
    var_1 = '{"key": "value", "number": 123, "bool": true}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'bool'
    var_5 = 'value'
    var_6 = 123
    var_7 = True
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = '{"outer": {"inner": "data"}}'
    var_10 = 'outer'
    var_11 = 'inner'
    var_12 = 'data'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = '{"key": "value",}'
    var_16 = '"just a string"'
    var_17 = '123'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Tests the JsonPrompt class properties and response processing.'
    var_1 = '{"key": "value", "number": 123}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'value'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '["item1", "item2"]'
    var_8 = '{"key": "value"'
    var_9 = None



# Parsed testcases at query #14
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_dict function for various input scenarios.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'test_var'
    var_5 = module_0.read_user_dict(var_4, var_3)
    var_6 = 'test_var'
    var_7 = 'Enter your config:'
    var_8 = {var_6: var_7}
    var_9 = '[bold]Prompt:[/]'
    var_10 = 'test_var'
    var_11 = module_0.read_user_dict(var_10, var_3, var_8, var_9)
    var_12 = '[bold]Prompt:[/][bold]Enter your config:[/] [cyan bold](default)[/]'
    var_13 = 'test_var'
    var_14 = 'not a dict'
    var_15 = module_0.read_user_dict(var_13, var_14)
    var_16 = 'test_var'
    var_17 = module_0.read_user_dict(var_16, var_3)
    var_18 = str(var_16)
    var_19 = 'test_var'
    var_20 = module_0.read_user_dict(var_19, var_3)
    var_21 = str(var_19)
    var_22 = 'unknown_var'
    var_23 = 'other'
    var_24 = 'val'
    var_25 = {var_23: var_24}
    var_26 = module_0.read_user_dict(var_22, var_3, var_25)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Test the read_user_dict function for various input scenarios.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'my_var [cyan bold](default)[/]'
    var_5 = 'default'
    var_6 = {var_5: var_3}
    var_7 = False
    var_8 = [var_4, var_6, var_7]
    var_9 = 'my_var'
    var_10 = module_0.read_user_dict(var_9, var_3)
    var_11 = 'my_var'
    var_12 = 'Custom Question'
    var_13 = {var_11: var_12}
    var_14 = 'PROMPT: '
    var_15 = 'a'
    var_16 = 1
    var_17 = {var_15: var_16}
    var_18 = 'PROMPT: Custom Question [cyan bold](default)[/]'
    var_19 = 'my_var'
    var_20 = module_0.read_user_dict(var_19, var_17, var_13, var_14)
    var_21 = 'my_var'
    var_22 = 'not a dict'
    var_23 = module_0.read_user_dict(var_21, var_22)
    var_24 = 'other_var'
    var_25 = 'Other'
    var_26 = {var_24: var_25}
    var_27 = {var_22: var_23}
    var_28 = 'my_var'
    var_29 = module_0.read_user_dict(var_28, var_27, var_26)
    var_30 = 'test_var'
    var_31 = 'default'
    var_32 = 'val'
    var_33 = {var_31: var_32}
    var_34 = module_0.read_user_dict(var_30, var_33)



# Parsed testcases at query #16
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_dict function for various scenarios.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'my_var'
    var_5 = module_0.read_user_dict(var_4, var_3)
    var_6 = 'my_var [cyan bold](default)[/]'
    var_7 = False
    var_8 = 'my_var'
    var_9 = 'Enter your configuration'
    var_10 = {var_8: var_9}
    var_11 = 'my_var'
    var_12 = module_0.read_user_dict(var_11, var_3, var_10)
    var_13 = 'Enter your configuration [cyan bold](default)[/]'
    var_14 = False
    var_15 = 'Config: '
    var_16 = 'my_var'
    var_17 = module_0.read_user_dict(var_16, var_3, prefix=var_15)
    var_18 = 'Config: my_var [cyan bold](default)[/]'
    var_19 = False
    var_20 = 'my_var'
    var_21 = 'not_a_dict'
    var_22 = module_0.read_user_dict(var_20, var_21)
    var_23 = '{invalid_json}'
    var_24 = module_0.process_json(var_23)
    var_25 = str(var_23)
    var_26 = '["not", "a", "dict"]'
    var_27 = module_0.process_json(var_26)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests the read_user_dict function with various inputs and mock responses.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 'old_value'
    var_4 = {var_1: var_3}
    var_5 = 'my_var'
    var_6 = module_0.read_user_dict(var_5, var_4)
    var_7 = 'my_var'
    var_8 = 'Custom Question'
    var_9 = {var_7: var_8}
    var_10 = 'Prefix: '
    var_11 = 'a'
    var_12 = 1
    var_13 = 'my_var'
    var_14 = 0
    var_15 = {var_11: var_14}
    var_16 = module_0.read_user_dict(var_13, var_15, var_9, var_10)
    var_17 = 'Prefix: Custom Question'
    var_18 = 'my_var'
    var_19 = 'not'
    var_20 = 'a'
    var_21 = 'dict'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.read_user_dict(var_18, var_22)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = "\n    Tests prompt_for_config by mocking the interactive input functions.\n    We simulate a 'no_input=True' scenario to avoid actual terminal interaction\n    and verify that variables are rendered and processed correctly.\n    "
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'repo_name'
    var_4 = 'is_active'
    var_5 = 'choices'
    var_6 = '__prompts__'
    var_7 = 'TestProject'
    var_8 = '{{ cookiecutter.project_name.lower().replace(" ", "_") }}'
    var_9 = True
    var_10 = 'alpha'
    var_11 = 'beta'
    var_12 = [var_10, var_11]
    var_13 = 'Name?'
    var_14 = {var_2: var_13}
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_12, var_6: var_14}
    var_16 = {var_1: var_15}
    var_17 = True
    var_18 = module_0.prompt_for_config(var_16, var_17)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests that UndefinedError in templates raises UndefinedVariableInTemplate.'
    var_1 = 'cookiecutter'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'value'
    var_5 = '{{ cookiecutter.non_existent }}'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = True
    var_9 = module_0.prompt_for_config(var_7, var_8)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests that dictionary variables are processed.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'settings'
    var_4 = 'Base'
    var_5 = 'env'
    var_6 = 'prod'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'env'
    var_11 = 'prod'
    var_12 = module_0.prompt_for_config(var_9, var_2)



# Parsed testcases at query #20
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
    var_12 = '["item1", "item2"]'
    var_13 = module_0.process_json(var_12)
    var_14 = '{"key": "value",}'
    var_15 = module_0.process_json(var_14)
    var_16 = ''
    var_17 = module_0.process_json(var_16)
    var_18 = '{"outer": {"inner": "val"}, "list": [1, 2]}'
    var_19 = 'outer'
    var_20 = 'inner'
    var_21 = 'val'
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = 'list'
    var_25 = 2
    var_26 = [var_8, var_25]
    var_27 = (var_24, var_26)
    var_28 = module_0.process_json(var_18)
    var_29 = '"just a string"'
    var_30 = module_0.process_json(var_29)



# Parsed testcases at query #21
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'var_name'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'my_value'
    var_3 = 'var_name'
    var_4 = 'Custom Question'
    var_5 = {var_3: var_4}
    var_6 = 'var_name'
    var_7 = 'default_val'
    var_8 = module_0.read_user_variable(var_6, var_7, var_5)
    assert var_8 == 'user_input'
    var_9 = 'Custom Question'
    var_10 = {var_6: var_7}
    var_11 = 'PROMPT: '
    var_12 = 'var_name'
    var_13 = 'default_val'
    var_14 = module_0.read_user_variable(var_12, var_13, var_10, var_11)
    assert var_14 == 'user_input'
    var_15 = 'PROMPT: Custom Question'
    var_16 = 'var_name'
    var_17 = 'default_val'
    var_18 = module_0.read_user_variable(var_16, var_17)
    assert var_18 == 'valid_response'
    var_19 = 'other_var'
    var_20 = 'Different Question'
    var_21 = {var_19: var_20}
    var_22 = 'var_name'
    var_23 = 'default_val'
    var_24 = module_0.read_user_variable(var_22, var_23, var_21)
    assert var_24 == 'val'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'my_var'
    var_1 = 'default_val'
    var_2 = module_0.read_user_variable(var_0, var_1)
    assert var_2 == 'default_val'
    var_3 = 'my_var'
    var_4 = 'What is your name?'
    var_5 = {var_3: var_4}
    var_6 = 'my_var'
    var_7 = 'default_val'
    var_8 = module_0.read_user_variable(var_6, var_7, var_5)
    assert var_8 == 'John Doe'
    var_9 = 'What is your name?'
    var_10 = 'my_var'
    var_11 = 'default_val'
    var_12 = 'Enter: '
    var_13 = module_0.read_user_variable(var_10, var_11, prefix=var_12)
    assert var_13 == 'Value'
    var_14 = 'Enter: my_var'
    var_15 = 'Name?'
    var_16 = {var_10: var_15}
    var_17 = 'my_var'
    var_18 = 'default_val'
    var_19 = 'SET '
    var_20 = module_0.read_user_variable(var_17, var_18, var_16, var_19)
    assert var_20 == 'John'
    var_21 = 'SET Name?'
    var_22 = 'var'
    var_23 = 'def'
    var_24 = module_0.read_user_variable(var_22, var_23)
    assert var_24 == 'first_input'
    var_25 = 'unknown_var'
    var_26 = 'def'
    var_27 = 'other'
    var_28 = 'desc'
    var_29 = {var_27: var_28}
    var_30 = module_0.read_user_variable(var_25, var_26, var_29)
    assert var_30 == 'Value'



# Parsed testcases at query #24
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
    var_6 = 'apple'
    var_7 = var_1.process_response(var_6)



# Parsed testcases at query #25
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
    var_12 = '{"outer": {"inner": "val"}}'
    var_13 = 'outer'
    var_14 = 'inner'
    var_15 = 'val'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = '{"key": "value",}'
    var_19 = module_0.process_json(var_18)
    var_20 = '["item1", "item2"]'
    var_21 = module_0.process_json(var_20)
    var_22 = '"just a string"'
    var_23 = module_0.process_json(var_22)
    var_24 = ''
    var_25 = module_0.process_json(var_24)



# Parsed testcases at query #26
#--------------------------


import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'Tests various scenarios for reading user variables.'
    var_1 = 'project_name'
    var_2 = 'default_val'
    var_3 = module_0.read_user_variable(var_1, var_2)
    assert var_3 == 'my_project'
    var_4 = 'project_name'
    var_5 = 'Enter your project name:'
    var_6 = {var_4: var_5}



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test the behavior and properties of the JsonPrompt class.'
    var_1 = '{"key": "value", "number": 123}'
    var_2 = 'key'
    var_3 = 'number'
    var_4 = 'value'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '["item1", "item2"]'
    var_8 = '{"key": "value"'
    var_9 = 'Enter JSON:'



