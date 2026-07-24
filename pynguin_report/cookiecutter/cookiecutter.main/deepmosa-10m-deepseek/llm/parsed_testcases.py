####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_cookiecutter_generate_context_with_valid_file. Retrieved 2/6 statements.
# Partially parsed test_cookiecutter_generate_context_with_invalid_file. Retrieved 3/7 statements.
# Partially parsed test_cookiecutter_get_user_config_custom_file. Retrieved 2/6 statements.
# Failed to parse test_cookiecutter_run_pre_prompt_hook_no_script.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)
    var_3 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)
    var_6 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice'
    var_3 = 'path'
    var_4 = 'nested_path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '.'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_0.prompt_for_config(var_4, var_5)
    var_7 = 'key'
    var_8 = bool('key' in var_6)
    assert var_8 is True
    var_9 = var_6['key']
    assert var_9 == 'default_value'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "Test Project"}'
    var_2 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": invalid}'
    var_2 = module_0.generate_context(var_0)
    var_3 = 'JSON decoding error'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.json'
    var_1 = '{"default_context": {"key": "value"}}'
    var_2 = 'default_context'

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'Option (path)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '.'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice'
    var_3 = 'path'
    var_4 = '/absolute/path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '.'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)
    var_12 = 'Illegal template path'



# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_cookiecutter_replay_with_bool_and_no_extra_context. Retrieved 15/27 statements.
# Partially parsed test_cookiecutter_replay_with_custom_path. Retrieved 15/27 statements.
# Partially parsed test_cookiecutter_with_nested_template_selection. Retrieved 15/31 statements.
# Partially parsed test_cookiecutter_cleanup_temp_dirs. Retrieved 11/31 statements.
# Partially parsed test_cookiecutter_with_default_config. Retrieved 13/25 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'You can not use both replay and no_input or extra_context at the same time.'
    var_1 = 'some_template'
    var_2 = True
    var_3 = module_0.cookiecutter(var_1, no_input=var_2, replay=var_2)
    var_4 = 'some_template'
    var_5 = True
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.cookiecutter(var_4, extra_context=var_8, replay=var_5)

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = '/fake/repo'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = lambda replay_dir, template_name: var_8
    var_10 = 'some_template'
    var_11 = True
    var_12 = False
    var_13 = None
    var_14 = module_1.cookiecutter(var_10, no_input=var_12, extra_context=var_13, replay=var_11)

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = '/fake/repo'
    var_3 = 'custom_template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = lambda replay_dir, template_name: var_8
    var_10 = 'some_template'
    var_11 = '/custom/replay.json'
    var_12 = False
    var_13 = None
    var_14 = module_1.cookiecutter(var_10, no_input=var_12, extra_context=var_13, replay=var_11)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice1'
    var_3 = 'path'
    var_4 = 'subdir'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = lambda context_file, default_context, extra_context: var_8
    var_10 = '/fake/repo/subdir'
    var_11 = lambda context, repo_dir, no_input: var_10
    var_12 = 'some_template'
    var_13 = True
    var_14 = module_0.cookiecutter(var_12, no_input=var_13)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/fake/temp/repo'
    var_1 = '/fake/base/repo'
    var_2 = True
    var_3 = (var_1, var_2)
    var_4 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_3
    var_5 = lambda repo_dir: var_0
    var_6 = None
    var_7 = lambda path: var_6
    var_8 = 'some_template'
    var_9 = True
    var_10 = module_0.cookiecutter(var_8, accept_hooks=var_9)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'abbreviations'
    var_1 = 'cookiecutters_dir'
    var_2 = 'default_context'
    var_3 = 'replay_dir'
    var_4 = {}
    var_5 = '/fake/dir'
    var_6 = {}
    var_7 = '/fake/replay'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = lambda config_file, default_config: var_8
    var_10 = 'some_template'
    var_11 = True
    var_12 = module_0.cookiecutter(var_10, default_config=var_11)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_with_path_object. Retrieved 1/5 statements.
# Partially parsed test_constructor_ensures_string_type. Retrieved 1/7 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/fake/string/path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == '/fake/string/path'

def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------




import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice'
    var_3 = 'path'
    var_4 = '/absolute/path'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/some/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice'
    var_3 = 'path'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/some/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice'
    var_3 = 'path'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '/some/repo'
    var_10 = True
    var_11 = module_0.choose_nested_template(var_8, var_9, var_10)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'name (/absolute/path)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/some/repo'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)

import cookiecutter.prompt as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'name ()'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '/some/repo'
    var_7 = True
    var_8 = module_0.choose_nested_template(var_5, var_6, var_7)



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_with_path_object. Retrieved 1/17 statements.


def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_path_object. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_path_object_containing_spaces. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/tmp/test'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == '/tmp/test'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == ''

def test_case_0():
    var_0 = '/tmp/test path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/tmp/test path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == '/tmp/test path'



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_cookiecutter_with_replay_bool. Retrieved 34/50 statements.
# Partially parsed test_cookiecutter_with_replay_string. Retrieved 34/50 statements.
# Partially parsed test_cookiecutter_without_replay. Retrieved 32/47 statements.
# Partially parsed test_cookiecutter_nested_template. Retrieved 25/33 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda replay_dir, template_name: var_18
    var_20 = 'default'
    var_21 = {var_15: var_20}
    var_22 = {var_14: var_21}
    var_23 = lambda context_file, default_context, extra_context: var_22
    var_24 = {}
    var_25 = lambda context, no_input: var_24
    var_26 = '/tmp/result'
    var_27 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_26
    var_28 = None
    var_29 = lambda replay_dir, template_name, context: var_28
    var_30 = {}
    var_31 = 'test'
    var_32 = True
    var_33 = module_0.cookiecutter(var_31, replay=var_32)
    assert var_33 == '/tmp/result'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda replay_dir, template_name: var_18
    var_20 = 'default'
    var_21 = {var_15: var_20}
    var_22 = {var_14: var_21}
    var_23 = lambda context_file, default_context, extra_context: var_22
    var_24 = {}
    var_25 = lambda context, no_input: var_24
    var_26 = '/tmp/result'
    var_27 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_26
    var_28 = None
    var_29 = lambda replay_dir, template_name, context: var_28
    var_30 = {}
    var_31 = 'test'
    var_32 = '/tmp/replay.json'
    var_33 = module_0.cookiecutter(var_31, replay=var_32)
    assert var_33 == '/tmp/result'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'default'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda context_file, default_context, extra_context: var_18
    var_20 = 'prompted'
    var_21 = {var_15: var_20}
    var_22 = lambda context, no_input: var_21
    var_23 = '/tmp/result'
    var_24 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_23
    var_25 = None
    var_26 = lambda replay_dir, template_name, context: var_25
    var_27 = {}
    var_28 = 'test'
    var_29 = 'extra'
    var_30 = {var_15: var_29}
    var_31 = module_0.cookiecutter(var_28, extra_context=var_30)
    assert var_31 == '/tmp/result'

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'template'
    var_16 = 'nested'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda context_file, default_context, extra_context: var_18
    var_20 = 'nested_template'
    var_21 = lambda context, repo_dir, no_input: var_20
    var_22 = '/tmp/nested_result'
    var_23 = lambda template, checkout, no_input, extra_context, replay, overwrite_if_exists, output_dir, config_file, default_config, password, directory, skip_if_file_exists, accept_hooks, keep_project_on_failure: var_22
    var_24 = {}



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_patch_import_path_for_repo_init_with_path. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/tmp/test_repo'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    var_3 = bool(var_1._repo_dir == var_0)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_cookiecutter_with_replay_bool. Retrieved 34/67 statements.
# Partially parsed test_cookiecutter_with_replay_file. Retrieved 34/67 statements.
# Partially parsed test_cookiecutter_without_replay. Retrieved 28/44 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)
    var_3 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)
    var_6 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda replay_dir, template_name: var_18
    var_20 = 'default'
    var_21 = {var_15: var_20}
    var_22 = {var_14: var_21}
    var_23 = lambda context_file, default_context, extra_context: var_22
    var_24 = {}
    var_25 = lambda context, no_input: var_24
    var_26 = '/tmp/project'
    var_27 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_26
    var_28 = None
    var_29 = lambda replay_dir, template_name, context: var_28
    var_30 = 'cookiecutter.main'
    var_31 = 'some_template'
    var_32 = True
    var_33 = module_0.cookiecutter(var_31, replay=var_32)
    assert var_33 == '/tmp/project'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda replay_dir, template_name: var_18
    var_20 = 'default'
    var_21 = {var_15: var_20}
    var_22 = {var_14: var_21}
    var_23 = lambda context_file, default_context, extra_context: var_22
    var_24 = {}
    var_25 = lambda context, no_input: var_24
    var_26 = '/tmp/project'
    var_27 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_26
    var_28 = None
    var_29 = lambda replay_dir, template_name, context: var_28
    var_30 = 'cookiecutter.main'
    var_31 = 'some_template'
    var_32 = '/tmp/replay.json'
    var_33 = module_0.cookiecutter(var_31, replay=var_32)
    assert var_33 == '/tmp/project'

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'default_context'
    var_2 = 'abbreviations'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {}
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_4}
    var_8 = lambda config_file=None, default_config=False: var_7
    var_9 = '/tmp/repo'
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_11
    var_13 = lambda repo_dir: repo_dir
    var_14 = 'cookiecutter'
    var_15 = 'key'
    var_16 = 'default'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = lambda context_file, default_context, extra_context: var_18
    var_20 = 'prompted'
    var_21 = {var_15: var_20}
    var_22 = lambda context, no_input: var_21
    var_23 = '/tmp/project'
    var_24 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_23
    var_25 = None
    var_26 = lambda replay_dir, template_name, context: var_25
    var_27 = 'cookiecutter.main'



# Parsed testcases at query #25
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_path_object. Retrieved 1/3 statements.
# Partially parsed test_constructor_path_object_converted_to_string. Retrieved 1/5 statements.
# Partially parsed test_constructor_string_remains_string. Retrieved 3/4 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == '/fake/path'

def test_case_0():
    var_0 = '/another/fake/path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/another/fake/path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_cookiecutter_cleanup_temp_repo. Retrieved 11/19 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'fake_template'
    var_5 = 'cookiecutter'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = lambda replay_dir, template_name: var_9
    var_11 = 'fake_template'
    var_12 = True
    var_13 = module_1.cookiecutter(var_11, replay=var_12)
    assert var_13 == '/fake/output'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'fake_template'
    var_5 = 'cookiecutter'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = lambda replay_dir, template_name: var_9
    var_11 = 'fake_template'
    var_12 = '/path/to/replay.json'
    var_13 = module_1.cookiecutter(var_11, replay=var_12)
    assert var_13 == '/fake/output'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'default'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'fake_template'
    var_10 = True
    var_11 = module_1.cookiecutter(var_9, no_input=var_10)
    assert var_11 == '/fake/output'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'default'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'fake_template'
    var_10 = 'extra'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = module_1.cookiecutter(var_9, extra_context=var_12)
    assert var_13 == '/fake/output'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'cookiecutter'
    var_5 = 'template'
    var_6 = 'nested_template'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'fake_template'
    var_10 = True
    var_11 = module_1.cookiecutter(var_9, no_input=var_10)
    assert var_11 == '/fake/nested_output'

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/temp_repo'
    var_4 = '/fake/base_repo'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'fake_template'
    var_9 = True
    var_10 = module_1.cookiecutter(var_8, no_input=var_9)

import cookiecutter.config as module_0
import cookiecutter.main as module_1

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = '/fake/repo'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'fake_template'
    var_8 = True
    var_9 = module_1.cookiecutter(var_7, no_input=var_8, keep_project_on_failure=var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = None
    var_3 = False
    var_4 = var_1 is not var_3
    var_5 = None
    var_6 = var_2 is not var_5
    var_7 = var_4 or var_6
    var_8 = var_0 and var_7
    assert var_8 is False

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = None
    var_3 = False
    var_4 = var_1 is not var_3
    var_5 = None
    var_6 = var_2 is not var_5
    var_7 = var_4 or var_6
    var_8 = var_0 and var_7
    assert var_8 is False

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = None
    var_3 = False
    var_4 = var_1 is not var_3
    var_5 = None
    var_6 = var_2 is not var_5
    var_7 = var_4 or var_6
    var_8 = var_0 and var_7
    assert var_8 is False

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = var_1 is not var_5
    var_7 = None
    var_8 = var_4 is not var_7
    var_9 = var_6 or var_8
    var_10 = var_0 and var_9
    assert var_10 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_cookiecutter_replay_bool_loads_from_replay_dir. Retrieved 32/40 statements.
# Partially parsed test_cookiecutter_replay_string_loads_from_specified_path. Retrieved 32/40 statements.
# Partially parsed test_cookiecutter_no_replay_generates_context_with_extra_context. Retrieved 29/36 statements.
# Partially parsed test_cookiecutter_nested_template_recursion. Retrieved 26/32 statements.
# Partially parsed test_cookiecutter_cleanup_temp_repo_dir. Retrieved 28/36 statements.
# Partially parsed test_cookiecutter_accept_hooks_false_skips_pre_prompt_hook. Retrieved 25/31 statements.
# Partially parsed test_cookiecutter_context_includes_template_and_output_dir. Retrieved 28/38 statements.
# Partially parsed test_cookiecutter_keep_project_on_failure_passes_to_generate_files. Retrieved 27/34 statements.
# Partially parsed test_cookiecutter_skip_if_file_exists_passes_to_generate_files. Retrieved 14/16 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.load'
    var_16 = 'cookiecutter'
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'cookiecutter.main.generate_context'
    var_22 = {}
    var_23 = {var_16: var_22}
    var_24 = 'cookiecutter.main.prompt_for_config'
    var_25 = {}
    var_26 = 'cookiecutter.main.dump'
    var_27 = 'cookiecutter.main.generate_files'
    var_28 = '/output'
    var_29 = 'template'
    var_30 = True
    var_31 = module_0.cookiecutter(var_29, replay=var_30)
    assert var_31 == '/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.load'
    var_16 = 'cookiecutter'
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'cookiecutter.main.generate_context'
    var_22 = {}
    var_23 = {var_16: var_22}
    var_24 = 'cookiecutter.main.prompt_for_config'
    var_25 = {}
    var_26 = 'cookiecutter.main.dump'
    var_27 = 'cookiecutter.main.generate_files'
    var_28 = '/output'
    var_29 = 'template'
    var_30 = '/custom/replay.json'
    var_31 = module_0.cookiecutter(var_29, replay=var_30)
    assert var_31 == '/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.generate_context'
    var_16 = 'cookiecutter'
    var_17 = 'extra_key'
    var_18 = 'extra_value'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'cookiecutter.main.prompt_for_config'
    var_22 = {}
    var_23 = 'cookiecutter.main.dump'
    var_24 = 'cookiecutter.main.generate_files'
    var_25 = '/output'
    var_26 = 'template'
    var_27 = {var_17: var_18}
    var_28 = module_0.cookiecutter(var_26, extra_context=var_27)
    assert var_28 == '/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.generate_context'
    var_16 = 'cookiecutter'
    var_17 = 'template'
    var_18 = 'nested'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'cookiecutter.main.choose_nested_template'
    var_22 = 'nested_template'
    var_23 = 'cookiecutter.main.cookiecutter'
    var_24 = '/nested_output'
    var_25 = module_0.cookiecutter(var_17)
    assert var_25 == '/nested_output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/base_repo'
    var_12 = True
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = '/temp_repo'
    var_16 = 'cookiecutter.main.generate_context'
    var_17 = 'cookiecutter'
    var_18 = {}
    var_19 = {var_17: var_18}
    var_20 = 'cookiecutter.main.prompt_for_config'
    var_21 = {}
    var_22 = 'cookiecutter.main.dump'
    var_23 = 'cookiecutter.main.generate_files'
    var_24 = '/output'
    var_25 = 'cookiecutter.main.rmtree'
    var_26 = 'template'
    var_27 = module_0.cookiecutter(var_26)
    assert var_27 == '/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.generate_context'
    var_15 = 'cookiecutter'
    var_16 = {}
    var_17 = {var_15: var_16}
    var_18 = 'cookiecutter.main.prompt_for_config'
    var_19 = {}
    var_20 = 'cookiecutter.main.dump'
    var_21 = 'cookiecutter.main.generate_files'
    var_22 = '/output'
    var_23 = 'template'
    var_24 = module_0.cookiecutter(var_23, accept_hooks=var_12)
    assert var_24 == '/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.generate_context'
    var_16 = 'cookiecutter'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = 'cookiecutter.main.prompt_for_config'
    var_20 = {}
    var_21 = 'cookiecutter.main.dump'
    var_22 = 'cookiecutter.main.generate_files'
    var_23 = '/output'
    var_24 = 'template_url'
    var_25 = '/custom_output'
    var_26 = module_0.cookiecutter(var_24, output_dir=var_25)
    var_27 = 2

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)
    var_14 = 'cookiecutter.main.run_pre_prompt_hook'
    var_15 = 'cookiecutter.main.generate_context'
    var_16 = 'cookiecutter'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = 'cookiecutter.main.prompt_for_config'
    var_20 = {}
    var_21 = 'cookiecutter.main.dump'
    var_22 = 'cookiecutter.main.generate_files'
    var_23 = '/output'
    var_24 = 'template'
    var_25 = True
    var_26 = module_0.cookiecutter(var_24, keep_project_on_failure=var_25)

def test_case_0():
    var_0 = 'cookiecutter.main.get_user_config'
    var_1 = 'abbreviations'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = '/replay'
    var_8 = {}
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter.main.determine_repo_dir'
    var_11 = '/repo'
    var_12 = False
    var_13 = (var_11, var_12)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_init_converts_path_to_string. Retrieved 1/9 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = None
    var_3 = bool(var_0 and (var_1 is not False or var_2 is not None))
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_cookiecutter_with_nested_template. Retrieved 38/49 statements.
# Partially parsed test_cookiecutter_with_replay. Retrieved 34/44 statements.
# Partially parsed test_cookiecutter_with_extra_context. Retrieved 31/40 statements.
# Partially parsed test_cookiecutter_cleanup_on_failure. Retrieved 28/42 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)
    var_3 = 'You can not use both replay and no_input or extra_context at the same time.'
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)
    var_6 = 'You can not use both replay and no_input or extra_context at the same time.'
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'choice1'
    var_3 = 'path'
    var_4 = 'subdir'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = lambda context_file, default_context, extra_context: var_8
    var_10 = lambda context, repo_dir, no_input: var_4
    var_11 = {}
    var_12 = lambda context_for_prompting, no_input: var_11
    var_13 = '/fake/output'
    var_14 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_13
    var_15 = 'abbreviations'
    var_16 = 'cookiecutters_dir'
    var_17 = 'replay_dir'
    var_18 = 'default_context'
    var_19 = {}
    var_20 = '/fake'
    var_21 = {}
    var_22 = {var_15: var_19, var_16: var_20, var_17: var_20, var_18: var_21}
    var_23 = lambda config_file, default_config: var_22
    var_24 = '/fake/repo'
    var_25 = False
    var_26 = (var_24, var_25)
    var_27 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_26
    var_28 = lambda repo_dir: repo_dir
    var_29 = {}
    var_30 = {var_0: var_29}
    var_31 = lambda replay_dir, template_name: var_30
    var_32 = None
    var_33 = lambda replay_dir, template_name, context: var_32
    var_34 = lambda path: var_32
    var_35 = 'template_with_nested'
    var_36 = True
    var_37 = module_0.cookiecutter(var_35, no_input=var_36)
    assert var_37 == '/fake/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda context_file, default_context, extra_context: var_4
    var_6 = {}
    var_7 = lambda context_for_prompting, no_input: var_6
    var_8 = '/fake/output'
    var_9 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_8
    var_10 = 'abbreviations'
    var_11 = 'cookiecutters_dir'
    var_12 = 'replay_dir'
    var_13 = 'default_context'
    var_14 = {}
    var_15 = '/fake'
    var_16 = {}
    var_17 = {var_10: var_14, var_11: var_15, var_12: var_15, var_13: var_16}
    var_18 = lambda config_file, default_config: var_17
    var_19 = '/fake/repo'
    var_20 = False
    var_21 = (var_19, var_20)
    var_22 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_21
    var_23 = lambda repo_dir: repo_dir
    var_24 = 'replay_value'
    var_25 = {var_1: var_24}
    var_26 = {var_0: var_25}
    var_27 = lambda replay_dir, template_name: var_26
    var_28 = None
    var_29 = lambda replay_dir, template_name, context: var_28
    var_30 = lambda path: var_28
    var_31 = 'some_template'
    var_32 = True
    var_33 = module_0.cookiecutter(var_31, replay=var_32)
    assert var_33 == '/fake/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda context_file, default_context, extra_context: var_4
    var_6 = {}
    var_7 = lambda context_for_prompting, no_input: var_6
    var_8 = '/fake/output'
    var_9 = lambda repo_dir, context, overwrite_if_exists, skip_if_file_exists, output_dir, accept_hooks, keep_project_on_failure: var_8
    var_10 = 'abbreviations'
    var_11 = 'cookiecutters_dir'
    var_12 = 'replay_dir'
    var_13 = 'default_context'
    var_14 = {}
    var_15 = '/fake'
    var_16 = {}
    var_17 = {var_10: var_14, var_11: var_15, var_12: var_15, var_13: var_16}
    var_18 = lambda config_file, default_config: var_17
    var_19 = '/fake/repo'
    var_20 = False
    var_21 = (var_19, var_20)
    var_22 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_21
    var_23 = lambda repo_dir: repo_dir
    var_24 = None
    var_25 = lambda replay_dir, template_name, context: var_24
    var_26 = lambda path: var_24
    var_27 = 'some_template'
    var_28 = 'overridden'
    var_29 = {var_1: var_28}
    var_30 = module_0.cookiecutter(var_27, extra_context=var_29)
    assert var_30 == '/fake/output'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = lambda context_file, default_context, extra_context: var_2
    var_4 = {}
    var_5 = lambda context_for_prompting, no_input: var_4
    var_6 = ()
    var_7 = 'Generation failed'
    var_8 = [var_7]
    var_9 = 'abbreviations'
    var_10 = 'cookiecutters_dir'
    var_11 = 'replay_dir'
    var_12 = 'default_context'
    var_13 = {}
    var_14 = '/fake'
    var_15 = {}
    var_16 = {var_9: var_13, var_10: var_14, var_11: var_14, var_12: var_15}
    var_17 = lambda config_file, default_config: var_16
    var_18 = '/fake/repo'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = lambda template, abbreviations, clone_to_dir, checkout, no_input, password, directory: var_20
    var_22 = lambda repo_dir: repo_dir
    var_23 = None
    var_24 = lambda replay_dir, template_name, context: var_23
    var_25 = lambda path: var_23
    var_26 = 'some_template'
    var_27 = False
    var_28 = module_0.cookiecutter(var_26, keep_project_on_failure=var_27)
    var_29 = bool(False)
    assert var_29 is True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = lambda context_file, default_context, extra_context: var_2
    var_4 = {}
    var_5 = lambda context_for_prompting, no_input: var_4



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_cookiecutter_replay_str_loads_from_path. Retrieved 12/14 statements.
# Partially parsed test_cookiecutter_nested_template_selection. Retrieved 13/16 statements.
# Partially parsed test_cookiecutter_prompt_for_config_integration. Retrieved 9/12 statements.
# Partially parsed test_cookiecutter_generate_files_called. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_cleanup_temp_dirs. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_context_includes_template. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_context_includes_output_dir. Retrieved 8/11 statements.
# Partially parsed test_cookiecutter_context_includes_repo_dir. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_context_includes_checkout. Retrieved 8/11 statements.
# Partially parsed test_cookiecutter_dump_replay_file. Retrieved 15/20 statements.
# Partially parsed test_cookiecutter_skip_if_file_exists. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_accept_hooks_false. Retrieved 8/11 statements.
# Partially parsed test_cookiecutter_keep_project_on_failure. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_with_directory_param. Retrieved 8/11 statements.
# Partially parsed test_cookiecutter_with_password. Retrieved 8/11 statements.
# Partially parsed test_cookiecutter_with_config_file. Retrieved 9/15 statements.
# Partially parsed test_cookiecutter_with_default_config. Retrieved 7/10 statements.
# Partially parsed test_cookiecutter_overwrite_if_exists. Retrieved 7/10 statements.


import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = module_0.cookiecutter(var_0, no_input=var_1, replay=var_1)
    var_3 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'some_template'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.cookiecutter(var_0, extra_context=var_4, replay=var_1)
    var_6 = 'You can not use both replay and no_input or extra_context at the same time.'

import cookiecutter.config as module_0
import cookiecutter.replay as module_1
import cookiecutter.main as module_2

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = 'replay_dir'
    var_4 = var_2[var_3]
    var_5 = 'test_template'
    var_6 = 'cookiecutter'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = module_1.dump(var_4, var_5, var_10)
    var_12 = 'test_template'
    var_13 = False
    var_14 = module_2.cookiecutter(var_12, no_input=var_13, replay=var_1)
    var_15 = bool(var_14 is not None)
    assert var_15 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/tmp/replay.json'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 0
    var_8 = os.path.splitext(var_0)[var_7]
    var_9 = 'test_template'
    var_10 = False
    var_11 = module_0.cookiecutter(var_9, no_input=var_10, replay=var_0)
    var_12 = bool(var_11 is not None)
    assert var_12 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = 'templates'
    var_3 = 'option1'
    var_4 = 'path'
    var_5 = 'subdir'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = '.'
    var_11 = True
    var_12 = module_0.cookiecutter(var_10, no_input=var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '.'
    var_7 = True
    var_8 = module_0.cookiecutter(var_6, no_input=var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'https://github.com/some/template.git'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '/tmp/output'
    var_5 = '.'
    var_6 = True
    var_7 = module_0.cookiecutter(var_5, no_input=var_6, output_dir=var_4)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'v1.0.0'
    var_5 = '.'
    var_6 = True
    var_7 = module_0.cookiecutter(var_5, var_4, var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import cookiecutter.config as module_0
import cookiecutter.main as module_1
import cookiecutter.replay as module_2

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = 'replay_dir'
    var_4 = var_2[var_3]
    var_5 = 'test_template'
    var_6 = 'cookiecutter.json'
    var_7 = 'cookiecutter'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = '.'
    var_13 = module_1.cookiecutter(var_12, no_input=var_1)
    var_14 = module_2.get_file_name(var_4, var_5)

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5, skip_if_file_exists=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = False
    var_6 = True
    var_7 = module_0.cookiecutter(var_4, no_input=var_6, accept_hooks=var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5, keep_project_on_failure=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = 'subdir'
    var_6 = True
    var_7 = module_0.cookiecutter(var_4, no_input=var_6, directory=var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = 'secret'
    var_6 = True
    var_7 = module_0.cookiecutter(var_4, no_input=var_6, password=var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'default_context:\n  project_name: Test'
    var_2 = 'cookiecutter.json'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = '.'
    var_7 = True
    var_8 = module_0.cookiecutter(var_6, no_input=var_7, config_file=var_0)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5, default_config=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True

import cookiecutter.main as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '.'
    var_5 = True
    var_6 = module_0.cookiecutter(var_4, no_input=var_5, overwrite_if_exists=var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_repo_dir_converted_to_string_when_path. Retrieved 1/8 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]



# Parsed testcases at query #18
#--------------------------




import cookiecutter.hooks as module_0

def test_case_0():
    var_0 = '/some/repo/dir'
    var_1 = False
    var_2 = module_0.run_pre_prompt_hook(var_0)
    var_3 = str(var_2)
    var_4 = var_3 if var_1 else var_0
    var_5 = bool(var_4 == var_0)
    assert var_5 is True
    var_6 = bool(var_4 == var_0)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_with_path_object. Retrieved 1/5 statements.
# Partially parsed test_constructor_path_object_converted_to_string. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/another/path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    assert var_2 == '/another/path'

def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]

import cookiecutter.main as module_0

def test_case_0():
    var_0 = '/string/path'
    var_1 = module_0._patch_import_path_for_repo(var_0)
    var_2 = var_1._repo_dir
    var_3 = bool(var_1._repo_dir == var_0)
    assert var_3 is True



