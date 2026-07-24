####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #4
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = None



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

# Partially parsed test_expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 2/3 statements.
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_env_var_set. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_env_var_not_set_user_config_exists. Retrieved 4/6 statements.


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

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 5/9 statements.
# Partially parsed test_get_user_config_with_no_config_file. Retrieved 1/3 statements.


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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay/path'
    var_5 = '/expanded/cookiecutters/path'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 4/9 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/test\ncookiecutters_dir: ~/test'
    var_2 = module_0.get_config(var_0)
    var_3 = '~/test'
    var_4 = var_2['replay_dir']
    var_5 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid yaml content'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- list item'
    var_2 = module_0.get_config(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key: value'
    var_2 = 'utf-8'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 6/11 statements.
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



# Parsed testcases at query #19
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
    var_2 = var_1['key']
    assert var_2 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['key']
    assert var_1 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['key']
    assert var_1 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #22
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/8 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/8 statements.
# Partially parsed test_get_config_merges_default_and_yaml_configs. Retrieved 14/24 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'non_dict.yaml'
    var_1 = '- list: item'
    var_2 = 'utf-8'

import yaml as module_0

def test_case_0():
    var_0 = 'abbreviations'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'custom_abbr'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '~/custom_replay'
    var_7 = '$HOME/custom_cookies'
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_9 = 'test_config.yaml'
    var_10 = {}
    var_11 = module_0.dump(var_8, **var_10)
    var_12 = 'utf-8'
    var_13 = 'custom_replay'
    var_14 = 'custom_cookies'



# Parsed testcases at query #24
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/invalid.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_yaml_dict_is_instance_of_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_config_with_valid_yaml_file. Retrieved 12/17 statements.


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
    var_9 = '$HOME/.replay'
    var_10 = '$HOME/.cookiecutters'
    var_11 = module_0.get_config(var_0)

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

# Partially parsed test_get_config_with_valid_path. Retrieved 4/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']

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



# Parsed testcases at query #29
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_yaml_error_raised_when_parsing_invalid_yaml. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'invalid: yaml: content: [unclosed'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: [[['
    var_1 = 'invalid.yaml'
    var_2 = module_0.get_config(var_1)



# Parsed testcases at query #32
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.safe_load(var_0)
    var_2 = {}
    var_3 = var_1 or var_2
    var_4 = bool(var_3 == {})
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay/path'
    var_5 = '/expanded/cookiecutters/path'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not a dictionary'



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_isinstance_check_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_config_path_does_not_exist. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/non/existent/path'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_config_returns_merged_dict_with_expanded_paths. Retrieved 7/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/test'
    var_4 = '$HOME/test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)

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
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.get_config(var_0)



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #41
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #42
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = '/expanded/replay'
    var_8 = '/expanded/cookies'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.get_config(var_0)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #44
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = '/expanded/path1'
    var_8 = '/expanded/path2'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.get_config(var_0)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #45
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay/path'
    var_5 = '/expanded/cookies/path'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 3/4 statements.


import yaml as module_0

def test_case_0():
    var_0 = None
    var_1 = 'dummy_path'
    var_2 = module_0.safe_load(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_yaml_dict_not_dict_type. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_get_config_returns_merged_config_with_expanded_paths. Retrieved 6/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_path.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml_path.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config_path.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'expected_replay_dir'
    var_4 = 'expected_cookies_dir'
    var_5 = module_0.get_config(var_0)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 3/5 statements.


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
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'path/to/env/config'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True

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



# Parsed testcases at query #51
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
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'



# Parsed testcases at query #53
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #55
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
    var_9 = '/expanded/replay/path'
    var_10 = '/expanded/cookiecutters/path'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_file.yaml'



# Parsed testcases at query #58
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'dummy_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
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



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #63
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 5/7 statements.


import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'test_config.yaml'
    var_2 = 'utf-8'
    var_3 = module_0.open(var_1, encoding=var_2)
    var_4 = module_1.safe_load(var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #64
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #66
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = '/expanded/replay_dir'
    var_8 = '/expanded/cookiecutters_dir'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0.get_config(var_0)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True

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



# Parsed testcases at query #67
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_yaml_dict_assignment. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = {}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 2/5 statements.
# Partially parsed test_get_user_config_with_no_config. Retrieved 1/3 statements.


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
    var_0 = '/custom/config'
    var_1 = module_0.get_user_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/path'
    var_3 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/path'
    var_2 = '/env/config'
    var_3 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/path'
    var_2 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/8 statements.
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



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_no_config_file. Retrieved 1/3 statements.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 2/3 statements.
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



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/8 statements.
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



# Parsed testcases at query #11
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/6 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_default_and_yaml_configs. Retrieved 9/13 statements.
# Partially parsed test_get_config_expands_environment_variables_in_paths. Retrieved 7/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = 'not_a_dict'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'new_key'
    var_3 = '~/custom_replay'
    var_4 = '~/custom_cookies'
    var_5 = 'new_value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'test_config.yaml'
    var_8 = module_0.get_config(var_7)
    var_9 = var_8['replay_dir']
    var_10 = var_8['cookiecutters_dir']
    var_11 = var_8['new_key']
    assert var_11 == 'new_value'
    var_12 = var_8['abbreviations']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '$TEST_DIR/replay'
    var_3 = '$TEST_DIR/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_config_env.yaml'
    var_6 = module_0.get_config(var_5)
    var_7 = var_6['replay_dir']
    assert var_7 == '/test/dir/replay'
    var_8 = var_6['cookiecutters_dir']
    assert var_8 == '/test/dir/cookies'



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_keyerror_raised_when_cookiecutter_config_not_set.




# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 6/10 statements.
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_keyerror_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #25
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



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/6 statements.


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



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #29
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 3/5 statements.


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
    var_0 = 'path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'path/to/env/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True

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



# Parsed testcases at query #31
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
    var_0 = '/custom/path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/test/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = 'Unable to parse YAML file'
    var_4 = bool('Unable to parse YAML file' in var_2)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------




import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = module_0.open(var_0)
    var_2 = module_1.safe_load(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'
    var_1 = 'key: value'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #36
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = ''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_config_returns_merged_default_and_yaml_config. Retrieved 10/12 statements.
# Partially parsed test_get_config_expands_paths_in_config. Retrieved 7/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n    key1: value1\n    nested:\n        key2: value2\n    '
    var_1 = 'dummy_path'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_2 == {'merged': 'config'})
    assert var_3 is True
    var_4 = 'key1'
    var_5 = 'nested'
    var_6 = 'value1'
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = {var_7: var_8}
    var_10 = {var_4: var_6, var_5: var_9}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_path'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml_path'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n    replay_dir: ~/replay\n    cookiecutters_dir: $HOME/cookiecutters\n    '
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/replay'
    var_4 = '$HOME/cookiecutters'
    var_5 = 'dummy_path'
    var_6 = module_0.get_config(var_5)
    var_7 = var_6['replay_dir']
    var_8 = var_6['cookiecutters_dir']



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_open_file_with_encoding. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'test: value'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


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
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_open_file_fails_when_file_does_not_exist. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.yaml'



# Parsed testcases at query #45
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_or_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file tests/invalid_yaml.yaml.'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_yaml_error_raised_when_parsing_fails. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #50
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/home/user/replay'
    var_3 = var_1['cookiecutters_dir']
    assert var_3 == '/home/user/cookiecutters'
    var_4 = var_1['default_context']
    var_5 = bool(var_1['default_context'] == {'key': 'value'})
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent/path.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/invalid.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/non_dict.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not a dictionary'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_no_config_found. Retrieved 1/3 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 2/3 statements.
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



# Parsed testcases at query #3
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay/path'
    var_5 = '/expanded/cookiecutters/path'
    var_6 = 'other_value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 7/14 statements.
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
    var_0 = 'custom_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.get_user_config(var_0)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.get_user_config()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'env_config.yaml'
    var_6 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.get_user_config()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/9 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/8 statements.
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 5/9 statements.
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



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_var_config. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/6 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #19
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_returns_merged_and_expanded_config. Retrieved 7/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$HOME/replays'
    var_4 = '~/templates'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/nonexistent.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/invalid.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/list.yaml'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.get_config(var_0)



# Parsed testcases at query #21
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
    var_9 = '/expanded/replay'
    var_10 = '/expanded/cookiecutters'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

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



# Parsed testcases at query #22
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
    var_9 = '/expanded/replay/path'
    var_10 = '/expanded/cookies/path'
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.get_config(var_0)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #24
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_yaml_dict_not_a_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not a dict'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_path'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #29
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
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 4/10 statements.
# Partially parsed test_get_config_with_nonexistent_path. Retrieved 1/4 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 2/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/templates'
    var_2 = '~/replays'
    var_3 = '~/templates'

def test_case_0():
    var_0 = 'nonexistent_config.yaml'

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- list item 1\n- list item 2'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_isinstance_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid_yaml_file.yaml.'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_yaml_safe_load_raises_yaml_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'invalid yaml content'



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml content'
    var_1 = module_0.safe_load(var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_config_returns_merged_and_expanded_config. Retrieved 5/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir: ~/test\ncookiecutters_dir: $HOME/test'
    var_2 = module_0.get_config(var_0)
    var_3 = '~/test'
    var_4 = var_2['replay_dir']
    var_5 = '$HOME/test'
    var_6 = var_2['cookiecutters_dir']



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 7/12 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/path'
    var_6 = module_0.get_config(var_0)
    var_7 = bool(var_6 == {'replay_dir': '/expanded_path', 'cookiecutters_dir': '/expanded_path'})
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid_yaml.yaml.'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_file.yaml'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 10/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/replays'
    var_4 = '$HOME/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'default_replays'
    var_7 = 'default_cookiecutters'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import yaml.error as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'Invalid YAML'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.YAMLError(*var_2, **var_3)
    var_5 = module_1.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.get_config(var_0)



# Parsed testcases at query #42
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_vars.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/home/user/replay'
    var_5 = '/home/user/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.get_config(var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True



# Parsed testcases at query #43
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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = ''
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 5/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/path'
    var_4 = module_0.get_config(var_0)
    var_5 = bool(var_4 == {'replay_dir': '/expanded_path', 'cookiecutters_dir': '/expanded_path'})
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_path'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
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
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(var_1 == {'key1': 'value1', 'key2': 'value2', 'nested': {'key3': 'value3'}})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']



