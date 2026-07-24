####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'custom_path'
    var_4 = module_0.get_user_config(var_3)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test__expand_path_with_user_home. Retrieved 2/3 statements.
# Partially parsed test__expand_path_with_both_environment_variable_and_user_home. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/$TEST_VAR'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~/test'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/normal/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/normal/path'



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_keyerror_raised_when_cookiecutter_config_not_set. Retrieved 1/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/test_invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_no_config. Retrieved 1/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 5/7 statements.


import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'valid_path.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = module_0.open(var_0, encoding=var_2)
    var_4 = module_1.safe_load(var_3)



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_yaml_error_raised_when_invalid_yaml. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = str(var_0)
    assert var_1 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '$HOME/replay'
    var_5 = '$HOME/cookiecutters'
    var_6 = 'value'
    var_7 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #15
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #16
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.safe_load(var_0)
    var_2 = {}
    var_3 = var_1 or var_2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_with_valid_yaml_file. Retrieved 8/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '$HOME/replay'
    var_5 = '$HOME/cookiecutters'
    var_6 = 'value'
    var_7 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '$HOME/replay'
    var_5 = '$HOME/cookiecutters'
    var_6 = 'other_value'
    var_7 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_user_config_path_exists.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid_yaml_file.yaml.'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_config_returns_merged_config_with_expanded_paths. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/test_replay'
    var_4 = '$HOME/test_cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/7 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 2/7 statements.
# Partially parsed test_get_config_merges_with_default_and_expands_paths. Retrieved 10/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid_top_level.yaml'
    var_1 = '- list item'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'new_key'
    var_4 = '$HOME/test_replay'
    var_5 = '~/test_cookies'
    var_6 = 'new_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = 'preserved_default_key'



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #27
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookiecutters'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #33
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #34
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 1/6 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 1/6 statements.
# Partially parsed test_get_config_merges_default_and_yaml_configs. Retrieved 7/14 statements.
# Partially parsed test_get_config_expands_paths_in_config. Retrieved 3/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid.yaml'

def test_case_0():
    var_0 = 'non_dict.yaml'

def test_case_0():
    var_0 = 'valid.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '~/replays'
    var_5 = '~/cookiecutters'
    var_6 = 'other_value'

def test_case_0():
    var_0 = 'valid.yaml'
    var_1 = '~/replays'
    var_2 = '~/cookiecutters'



# Parsed testcases at query #36
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_file.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_or_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #40
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'nested_key'
    var_7 = 'nested_value'
    var_8 = {var_6: var_7}
    var_9 = '/expanded/replay_dir'
    var_10 = '/expanded/cookiecutters_dir'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_yaml_dict_is_dict_when_valid. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #42
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '~/.replay'
    var_5 = '~/.cookiecutters'
    var_6 = 'value'
    var_7 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #48
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_yaml_error_raised_when_parsing_invalid_yaml. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['



# Parsed testcases at query #50
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_empty_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_file.yaml'



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_yaml_safe_load_returns_none.




# Parsed testcases at query #54
#--------------------------

# Partially parsed test_config_path_exists_and_is_openable. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_yaml_dict_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_predicate_at_line_8_evaluates_to_true.




# Parsed testcases at query #59
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/10 statements.
# Partially parsed test_get_config_with_nonexistent_path. Retrieved 1/4 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 1/4 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}

def test_case_0():
    var_0 = 'nonexistent_config.yaml'

def test_case_0():
    var_0 = 'invalid_yaml.yaml'

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_no_config. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/custom/config'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/env/config'
    var_1 = module_0.get_user_config()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #61
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #63
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not a dict'



# Parsed testcases at query #65
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'nested_key'
    var_7 = 'nested_value'
    var_8 = {var_6: var_7}
    var_9 = '/expanded/replay_dir'
    var_10 = '/expanded/cookiecutters_dir'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #66
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/invalid.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #69
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #71
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'nested_key'
    var_7 = 'nested_value'
    var_8 = {var_6: var_7}
    var_9 = '/expanded/path1'
    var_10 = '/expanded/path2'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/9 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = ''



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid_yaml_file.yaml.'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_get_config_with_valid_file. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = '$HOME/cookiecutters'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_config_path.yaml'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 3/4 statements.


import yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'dummy'
    var_2 = module_0.safe_load(var_1)
    assert var_2 is None



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [[['
    var_1 = 'test_config.yaml'
    var_2 = module_0.get_config(var_1)



# Parsed testcases at query #83
#--------------------------

# Failed to parse test_yaml_safe_load_returns_none.




# Parsed testcases at query #84
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #85
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #88
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay_dir'
    var_5 = '/expanded/cookiecutters_dir'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_yaml_error_raised_when_parsing_fails. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed: bracket'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = '$HOME/cookiecutters'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_key'
    var_1 = 'test_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'test_key'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'test_key'
    var_2 = 'test_value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'test_key'
    var_2 = 'test_value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_user_home. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_both_expansions. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/$TEST_VAR'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~/test'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._expand_path(var_0)
    assert var_1 == ''

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'relative/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == 'relative/path'



# Parsed testcases at query #3
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay/path'
    var_5 = '/expanded/cookies/path'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/valid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/valid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = '$HOME/cookiecutters'



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 5/7 statements.


import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'empty.yaml'
    var_2 = 'utf-8'
    var_3 = module_0.open(var_1, encoding=var_2)
    var_4 = module_1.safe_load(var_3)



# Parsed testcases at query #11
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'
    var_1 = module_0.safe_load(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 2/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/path'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'env_path'
    var_5 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #19
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_config_path'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/6 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_with_default_config. Retrieved 8/13 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 7/11 statements.
# Partially parsed test_get_config_preserves_nested_dicts. Retrieved 10/12 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict.yaml'
    var_1 = '- list item'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'new_key'
    var_3 = '~/.test_replay'
    var_4 = 'new_value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)
    var_7 = 'cookiecutters_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_env.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$HOME/test_replay'
    var_4 = '$USER/test_cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_nested.yaml'
    var_1 = 'abbreviations'
    var_2 = 'test'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = module_0.get_config(var_0)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/7 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_default_and_expands_paths. Retrieved 10/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = 'invalid: yaml: content: [unclosed'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = '- list item'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'new_key'
    var_3 = '~/test_replay'
    var_4 = '$TEST_DIR/cookiecutters'
    var_5 = 'new_value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'test_config.yaml'
    var_8 = module_0.get_config(var_7)
    var_9 = 'default_key'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_user_home. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_both_expansions. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/test'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~/test'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._expand_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #7
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_keyerror_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_no_config. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'env_path'
    var_5 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #17
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_keyerror_predicate.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 2/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_keyerror_predicate.




# Parsed testcases at query #24
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = True
    var_2 = 'key: value'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 5/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_default_path. Retrieved 5/9 statements.
# Partially parsed test_get_user_config_with_no_config. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_without_env_var_and_user_config_exists. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_without_env_var_and_user_config_not_exists. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/custom/config'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/env/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



