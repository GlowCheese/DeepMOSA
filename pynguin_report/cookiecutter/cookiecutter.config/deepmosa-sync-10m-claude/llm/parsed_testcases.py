####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_env_variable_set. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_no_config_file_exists. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 7/9 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/8 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 5/10 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 7/13 statements.
# Partially parsed test_get_user_config_default_when_no_config_exists. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_env_variable_takes_precedence. Retrieved 7/14 statements.
# Partially parsed test_get_user_config_config_file_param_takes_precedence. Retrieved 6/14 statements.
# Partially parsed test_get_user_config_invalid_config_file_raises_error. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_invalid_yaml_raises_error. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_default_config_dict_priority. Retrieved 6/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config with default_config as a dict.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = ''
    var_3 = 'replay_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = var_6['replay_dir']
    assert var_7 == '/custom/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config with default_config as True.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = ''
    var_3 = True
    var_4 = module_0.get_user_config(default_config=var_3)

def test_case_0():
    var_0 = 'Test get_user_config with a custom config file path.'
    var_1 = 'custom_config.yaml'
    var_2 = 'replay_dir: /custom/replay\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config with COOKIECUTTER_CONFIG environment variable.'
    var_1 = 'env_config.yaml'
    var_2 = 'replay_dir: /env/replay\n'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = module_0.get_user_config()
    var_5 = var_4['replay_dir']
    assert var_5 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config when USER_CONFIG_PATH exists.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = 'user_config.yaml'
    var_4 = 'replay_dir: /user/replay\n'
    var_5 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_6 = module_0.get_user_config()
    var_7 = var_6['replay_dir']
    assert var_7 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns default when no config exists.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_4 = 'nonexistent.yaml'
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test that COOKIECUTTER_CONFIG env variable takes precedence.'
    var_1 = 'env_config.yaml'
    var_2 = 'replay_dir: /env/replay\n'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_5 = 'nonexistent.yaml'
    var_6 = module_0.get_user_config()
    var_7 = var_6['replay_dir']
    assert var_7 == '/env/replay'

def test_case_0():
    var_0 = 'Test that config_file parameter takes precedence over env variable.'
    var_1 = 'custom_config.yaml'
    var_2 = 'replay_dir: /custom/replay\n'
    var_3 = 'env_config.yaml'
    var_4 = 'replay_dir: /env/replay\n'
    var_5 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test that invalid config file path raises ConfigDoesNotExistException.'
    var_1 = 'nonexistent.yaml'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test that invalid YAML raises InvalidConfiguration.'
    var_1 = 'invalid.yaml'
    var_2 = '{ invalid yaml content: ['
    var_3 = module_0.get_user_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'Test that default_config dict has priority over config_file.'
    var_1 = 'custom_config.yaml'
    var_2 = 'replay_dir: /custom/replay\n'
    var_3 = 'replay_dir'
    var_4 = '/override/replay'
    var_5 = {var_3: var_4}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_invalid_env_variable. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_with_no_env_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 6/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_user_config()
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = module_0.get_user_config()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_valid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 2/12 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 4/17 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 1/8 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 1/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '- item1\n- item2\n'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = 'TEST_REPLAY_DIR'

def test_case_0():
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_1 = 'replay_dir'
    var_2 = '~'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: /custom/replays\n'
    var_1 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: /replays\ncookiecutters_dir: /cookies\nabbreviations:\n  custom_key: custom_value\n'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_raises_when_config_path_does_not_exist. Retrieved 2/5 statements.


def test_case_0():
    var_0 = "Test that get_config raises ConfigDoesNotExistException when config file doesn't exist."
    var_1 = 'non_existent_config.yaml'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 4/11 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 4/7 statements.
# Partially parsed test_expand_path_with_both_variables_and_home. Retrieved 4/11 statements.
# Partially parsed test_expand_path_with_multiple_environment_variables. Retrieved 6/19 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'TEST_VAR'
    var_1 = '$TEST_VAR/file.txt'
    var_2 = module_0._expand_path(var_1)
    assert var_2 == '/test/path/file.txt'
    var_3 = 'TEST_VAR'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/documents/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'
    var_3 = 'documents/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'HOME_VAR'
    var_1 = '$HOME_VAR/~/file.txt'
    var_2 = module_0._expand_path(var_1)
    var_3 = '/home/user'
    var_4 = bool('/home/user' in var_2)
    assert var_4 is True
    var_5 = 'HOME_VAR'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'VAR1'
    var_1 = 'VAR2'
    var_2 = '$VAR1/$VAR2/file.txt'
    var_3 = module_0._expand_path(var_2)
    assert var_3 == '/path1/path2/file.txt'
    var_4 = 'VAR1'
    var_5 = 'VAR2'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$NONEXISTENT_VAR_12345/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = bool('$NONEXISTENT_VAR_12345' in var_1 or var_1 == '$NONEXISTENT_VAR_12345/file.txt')
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_with_existing_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_file_does_not_exist. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_config_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_env_variable_set. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_env_variable_set_nonexistent_file. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_no_env_variable_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_config_file_parameter_takes_precedence. Retrieved 4/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /env/replays\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replays'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = '/nonexistent/env/config.yaml'
    var_2 = module_0.get_user_config()
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookies'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /file/replays\ncookiecutters_dir: /file/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = '/env/config.yaml'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 4/10 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_default_path_exists. Retrieved 8/17 statements.
# Partially parsed test_get_user_config_default_path_not_exists. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_config_file_takes_precedence_over_default_path. Retrieved 4/10 statements.
# Partially parsed test_get_user_config_default_config_dict_takes_precedence. Retrieved 7/13 statements.
# Partially parsed test_get_user_config_merges_nested_dicts. Retrieved 6/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'replay_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_3)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /default/replay\ncookiecutters_dir: /default/cookies'
    var_2 = 'os.path.exists'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = '__main__.USER_CONFIG_PATH'
    var_6 = None
    var_7 = module_0.get_user_config(var_6)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = ''
    var_2 = 'os.path.exists'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = module_0.get_user_config()

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'replay_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /file/replay'
    var_2 = 'replay_dir'
    var_3 = '/override/replay'
    var_4 = {var_2: var_3}
    var_5 = 'COOKIECUTTER_CONFIG'
    var_6 = False

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'abbreviations'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'abbreviations'
    var_7 = bool('abbreviations' in var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 5/6 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 4/11 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 6/14 statements.
# Partially parsed test_get_user_config_default_when_no_env_or_file. Retrieved 6/9 statements.
# Partially parsed test_get_user_config_prefers_default_config_dict_over_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_prefers_default_config_true_over_file. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns merged config when default_config is a dict.'
    var_1 = 'replay_dir'
    var_2 = '/custom/path'
    var_3 = {var_1: var_2}
    var_4 = module_0.get_user_config(default_config=var_3)
    var_5 = var_4['replay_dir']
    assert var_5 == '/custom/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns default config when default_config is True.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)

def test_case_0():
    var_0 = 'Test get_user_config loads custom config file when specified.'
    var_1 = 'custom_config.yaml'
    var_2 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies'
    var_3 = 'replay_dir'
    var_4 = '/tmp/replays'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config loads config from COOKIECUTTER_CONFIG environment variable.'
    var_1 = 'env_config.yaml'
    var_2 = 'replay_dir: /env/replays\ncookiecutters_dir: /env/cookies'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns default config when no env var or file exists.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = 'os.path.exists'
    var_4 = lambda x: var_2
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config raises exception for nonexistent custom config file.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_user_config(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test get_user_config prefers default_config dict over config_file.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /file/replays'
    var_3 = 'replay_dir'
    var_4 = '/dict/path'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test get_user_config prefers default_config=True over config_file.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /file/replays'
    var_3 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 7/22 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = False
    var_5 = bool(var_2 == {})
    assert var_5 is True
    var_6 = ''
    var_7 = False
    var_8 = module_0.get_user_config(var_6, var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 6/15 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = 'USER_CONFIG_PATH'
    var_5 = False
    var_6 = module_0.get_user_config(var_4, var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 11/22 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = 'config'
    var_5 = {var_3: var_4}
    var_6 = 'user_config.json'
    var_7 = '{}'
    var_8 = None
    var_9 = False
    var_10 = module_0.get_user_config(var_8, var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 6/12 statements.
# Partially parsed test_get_user_config_without_env_variable_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_default_config_false_and_no_config_file. Retrieved 5/8 statements.
# Partially parsed test_expand_path_with_environment_variables. Retrieved 4/5 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'replay_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: [yaml: content:'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'HOME'
    var_4 = False
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'HOME'
    var_3 = '/nonexistent/home'
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'HOME'
    var_3 = '/nonexistent/home'
    var_4 = module_0.get_user_config(default_config=var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'TEST_VAR'
    var_1 = '/test/path'
    var_2 = '$TEST_VAR/config'
    var_3 = module_0._expand_path(var_2)
    assert var_3 == '/test/path/config'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'HOME'
    var_1 = '/home/testuser'
    var_2 = '~/config'
    var_3 = module_0._expand_path(var_2)
    assert var_3 == '/home/testuser/config'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value2'
    var_6 = {var_1: var_5}
    var_7 = module_0.merge_configs(var_4, var_6)
    var_8 = var_7['key1']
    assert var_8 == 'value1'
    var_9 = var_7['key2']
    assert var_9 == 'new_value2'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner1'
    var_2 = 'inner2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.merge_configs(var_6, var_9)
    var_11 = var_10['outer']['inner1']
    assert var_11 == 'value1'
    var_12 = var_10['outer']['inner2']
    assert var_12 == 'new_value2'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = module_0.merge_configs(var_2, var_5)
    var_7 = var_6['key1']
    assert var_7 == 'value1'
    var_8 = var_6['key2']
    assert var_8 == 'value2'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = 'original'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'modified'
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.merge_configs(var_6, var_10)
    var_12 = var_11['level1']['level2']['level3']
    assert var_12 == 'modified'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value2'
    var_6 = {var_1: var_5}
    var_7 = module_0.merge_configs(var_4, var_6)
    var_8 = bool(var_4 == {'key1': 'value1', 'key2': 'value2'})
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = {var_0: var_1}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = '/home/user/.cookiecutterrc'
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = 'COOKIECUTTER_CONFIG'
    var_6 = False
    var_7 = True
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 7/13 statements.
# Partially parsed test_get_config_with_tilde_expansion. Retrieved 3/7 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 3/7 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/home/user/replays'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/home/user/cookies'
    var_4 = 'config.yaml'
    var_5 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n'
    var_6 = 'utf-8'
    var_7 = '/home/user/replays'
    var_8 = '/home/user/cookies'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'abbreviations:\n  custom_key: custom_value\nreplay_dir: /tmp\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'
    var_4 = 'custom_key'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 6/12 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 7/14 statements.
# Partially parsed test_get_user_config_default_when_no_config_exists. Retrieved 7/11 statements.
# Partially parsed test_get_user_config_invalid_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_config_file_takes_precedence. Retrieved 4/10 statements.
# Partially parsed test_get_user_config_merges_with_defaults. Retrieved 2/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replays'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'HOME'
    var_4 = False
    var_5 = module_0.get_user_config()
    var_6 = 'replay_dir'
    var_7 = bool('replay_dir' in var_5)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replays'
    var_2 = 'os.path.exists'
    var_3 = 'user_config'
    var_4 = False
    var_5 = 'COOKIECUTTER_CONFIG'
    var_6 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'HOME'
    var_1 = '/nonexistent'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = lambda x: var_3
    var_6 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /custom/path'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = '/env/config.yaml'

def test_case_0():
    var_0 = 'partial_config.yaml'
    var_1 = 'replay_dir: /custom/replay'
    var_2 = 'cookiecutters_dir'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false_and_no_env_and_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 9/15 statements.
# Partially parsed test_get_user_config_config_file_precedence_over_default. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = bool('replay_dir' in var_1)
    assert var_3 is True
    var_4 = 'cookiecutters_dir'
    var_5 = bool('cookiecutters_dir' in var_1)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config(default_config=var_1)
    var_5 = 'replay_dir'
    var_6 = bool('replay_dir' in var_4)
    assert var_6 is True

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/env/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'builtins.open'
    var_8 = module_0.get_user_config()
    var_9 = var_8['replay_dir']
    assert var_9 == '/user/replay'
    var_10 = var_8['cookiecutters_dir']
    assert var_10 == '/user/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'other_key'
    var_2 = '/custom'
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom'
    var_7 = var_5['other_key']
    assert var_7 == 'value'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /file/replay\n'
    var_2 = False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 8/25 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = 'value'
    var_5 = 'default'
    var_6 = 'config'
    var_7 = module_0.get_user_config()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 9/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'mocked'
    var_1 = 'config'
    var_2 = None
    var_3 = False
    var_4 = module_0.get_user_config(var_2, var_3)
    var_5 = bool(var_4 == {'mocked': 'config'})
    assert var_5 is True
    var_6 = '/home/user/.cookiecutterrc'
    var_7 = module_0.get_user_config(var_6, var_3)
    var_8 = bool(var_7 == {'mocked': 'config'})
    assert var_8 is True
    var_9 = ''
    var_10 = module_0.get_user_config(var_9, var_3)
    var_11 = bool(var_10 == {'mocked': 'config'})
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 7/21 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test: value'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = None
    var_5 = False
    var_6 = module_0.get_user_config(var_4, var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_predicate_at_line_3_evaluates_to_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True when config file exists.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_custom_file_takes_precedence. Retrieved 4/12 statements.
# Partially parsed test_get_user_config_from_environment_variable. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_default_when_no_env_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_invalid_config_file. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir: /test/replay\ncookiecutters_dir: /test/cookies'
    var_2 = '/test/replay'

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/path'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'env_config.yaml'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = '{ invalid yaml content ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/override/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/override/replay'
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_3)
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = module_0.get_user_config(default_config=var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True
    var_4 = bool(var_1 is not var_2)
    assert var_4 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 3/11 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/10 statements.
# Partially parsed test_get_user_config_prioritizes_default_config_dict_over_file. Retrieved 5/12 statements.
# Partially parsed test_get_user_config_prioritizes_default_config_true_over_file. Retrieved 3/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir: /test/replay\n'
    var_2 = False

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /custom/path\n'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /file/path\n'
    var_2 = 'replay_dir'
    var_3 = '/dict/path'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /file/path\n'
    var_2 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'some_value'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = True
    assert var_5 is False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 7/19 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = False
    var_5 = ''
    var_6 = False
    var_7 = module_0.get_user_config(var_5, var_6)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_path_does_not_exist. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that get_config raises ConfigDoesNotExistException when config file does not exist.'
    var_1 = 'non_existent_config.yaml'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 4/20 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config()
    var_4 = bool(var_3 == var_2)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'user_config.yaml'
    var_7 = 'test: config'
    var_8 = 'COOKIECUTTER_CONFIG'
    var_9 = False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_path_does_not_exist. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'Test that get_config raises ConfigDoesNotExistException when config file does not exist.'
    var_1 = 'non_existent_config.yaml'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 13/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = '/default/path/config'
    var_4 = 'default'
    var_5 = 'config'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = module_0.get_user_config(var_7, var_8)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True
    var_11 = module_0.get_user_config(var_3, var_8)
    var_12 = bool(var_11 == var_6)
    assert var_12 is True
    var_13 = ''
    var_14 = module_0.get_user_config(var_13, var_8)
    var_15 = bool(var_14 == var_6)
    assert var_15 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_default_config_true_takes_precedence. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_returns_dict. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_file. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = bool('replay_dir' in var_1)
    assert var_3 is True
    var_4 = 'cookiecutters_dir'
    var_5 = bool('cookiecutters_dir' in var_1)
    assert var_5 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'replay_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/priority/replay'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = module_0.get_user_config(var_3, var_2)
    var_5 = var_4['replay_dir']
    assert var_5 == '/priority/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = 'replay_dir'
    var_4 = bool('replay_dir' in var_2)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_3)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 3/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_default_path_exists. Retrieved 8/15 statements.
# Partially parsed test_get_user_config_no_config_file. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_default_config_false_priority. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_preserves_defaults_when_dict_provided. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()
    var_5 = var_4['replay_dir']
    assert var_5 == '/env/replay'

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'user_config.yaml'
    var_3 = 'replay_dir: /user/replay\n'
    var_4 = 'os.path.exists'
    var_5 = 'builtins.open'
    var_6 = None
    var_7 = lambda *args, **kwargs: var_6

import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'templates_dir'
    var_2 = '/custom/path'
    var_3 = '/custom/templates'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/path'
    var_7 = var_5['templates_dir']
    assert var_7 == '/custom/templates'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_returns_dict. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_false_default_config_no_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_with_env_var_nonexistent_file. Retrieved 5/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/path'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/path'
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_3)
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'HOME'
    var_3 = module_0.get_user_config(default_config=var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = None
    var_5 = module_0.get_user_config(var_4, var_3)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = '/nonexistent/env/config.yaml'
    var_2 = False
    var_3 = None
    var_4 = module_0.get_user_config(var_3, var_2)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = module_0.get_user_config(default_config=var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True
    var_4 = bool(var_1 is not var_2)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 3/17 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = module_0.get_user_config()
    var_3 = bool(var_2 == {'test': 'config'})
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 7/12 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 6/15 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 3/7 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 4/9 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/custom/replay'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/custom/cookies'
    var_4 = 'config.yaml'
    var_5 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n'
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replay\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'
    var_5 = 'replay_dir'
    var_6 = '~'
    var_7 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  custom: value\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 9/25 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True
    var_4 = 'key'
    var_5 = 'value'
    var_6 = False
    var_7 = bool(var_2 == {'key': 'value'})
    assert var_7 is True
    var_8 = ''
    var_9 = False
    var_10 = module_0.get_user_config(var_8, var_9)
    var_11 = bool(var_10 == {})
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_config_raises_when_config_path_does_not_exist. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Test that get_config raises ConfigDoesNotExistException when config file does not exist.'
    var_1 = 'non_existent_config.yaml'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_default_fallback. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_nonexistent_config_file_raises_error. Retrieved 1/5 statements.
# Partially parsed test_get_user_config_invalid_yaml_raises_error. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_non_dict_yaml_raises_error. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutterrc'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = ''
    var_4 = False
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = None
    var_3 = module_0.get_user_config(var_2, var_1)

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/path'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/path'
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_3)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_evaluates_to_empty_dict. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = {}
    var_3 = var_0 or var_2
    var_4 = bool(var_3 == {})
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 4/8 statements.
# Partially parsed test_get_config_with_environment_variables. Retrieved 5/14 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = 'replays'
    var_2 = 'config.yaml'
    var_3 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'
    var_4 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_yaml_error_predicate_evaluates_to_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'Test that the except clause at line 11 does NOT catch non-YAML errors.'
    var_1 = 'config.yaml'
    var_2 = 'valid: yaml\ncontent: here'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_config_opens_file_with_utf8_encoding. Retrieved 4/18 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp'
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 14 evaluates to False when yaml_dict is not a dict.'
    var_1 = 'config.yaml'
    var_2 = '- item1\n- item2\n'
    var_3 = 'utf-8'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/7 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 4/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'abbreviations:\n  custom_key: custom_value\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'
    var_4 = 'abbreviations'



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = '/default/config/path'
    var_1 = '/default/config/path'
    var_2 = var_1 is not var_0
    var_3 = var_1 and var_2
    assert var_3 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unable to parse YAML file'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_2 = 'utf-8'
    var_3 = {}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_line_40_predicate_evaluates_to_false. Retrieved 10/19 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = 'key'
    var_5 = 'default_value'
    var_6 = {var_4: var_5}
    var_7 = '/home/user/.cookiecutterrc'
    var_8 = None
    var_9 = module_0.get_user_config(var_8, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_config_predicate_line_8_evaluates_to_false. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\nreplay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'nonexistent.yaml'
    var_3 = True
    var_4 = False
    assert var_4 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_not_dict_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_valid_yaml_with_expansion. Retrieved 4/16 statements.
# Partially parsed test_get_config_custom_values_merged. Retrieved 1/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '- item1\n- item2\n'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: $HOME/.cookiecutters\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = '~'
    var_3 = '$'
    var_4 = 'replay_dir'
    var_5 = '/'
    var_6 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: /custom/replays\ncookiecutters_dir: /custom/cookies\n'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 7/42 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/default'
    var_6 = {var_3: var_5, var_4: var_5}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_expand_path_with_env_variable. Retrieved 2/4 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 2/4 statements.
# Partially parsed test_expand_path_with_both_env_and_home. Retrieved 4/8 statements.
# Partially parsed test_expand_path_with_tilde_only. Retrieved 2/4 statements.
# Partially parsed test_expand_path_with_multiple_env_variables. Retrieved 2/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/documents/file.txt'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~'
    var_1 = '$HOME_DIR/documents/file.txt'
    var_2 = module_0._expand_path(var_1)
    var_3 = '/documents/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$DIR1/$DIR2/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/dir1/dir2/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = './relative/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == './relative/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$NONEXISTENT_VAR_XYZ/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '$NONEXISTENT_VAR_XYZ/file.txt'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_config_predicate_line_8_evaluates_to_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 2/6 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 2/5 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 2/5 statements.
# Partially parsed test_get_config_with_path_object. Retrieved 2/6 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 2/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: $HOME/replays\ncookiecutters_dir: $HOME/cookies\n'
    var_2 = '$HOME'
    var_3 = '$HOME'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'replay_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = '~'
    var_3 = '~'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'config'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 5/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'mocked'
    var_1 = 'default_config'
    var_2 = None
    var_3 = False
    var_4 = module_0.get_user_config(var_2, var_3)
    var_5 = bool(var_4 == {'mocked': 'default_config'})
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/10 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/10 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 1/9 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 1/8 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 1/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = '/nonexistent/path/to/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = True
    assert var_3 is True

def test_case_0():
    var_0 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = None
    var_2 = True
    assert var_2 is True

def test_case_0():
    var_0 = '- item1\n- item2\n'
    var_1 = None
    var_2 = True
    assert var_2 is True

def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = '/test/replays'

def test_case_0():
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_1 = '~'
    var_2 = '~'

def test_case_0():
    var_0 = 'replay_dir: /custom/replays\ncookiecutters_dir: /custom/cookies\n'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 1/11 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 2/13 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 4/16 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 1/9 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/10 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies'
    var_1 = 'TEST_REPLAY_DIR'

def test_case_0():
    var_0 = 'replay_dir: ~/replay\ncookiecutters_dir: ~/cookies'
    var_1 = 'replay_dir'
    var_2 = '~'
    var_3 = 'cookiecutters_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True

def test_case_0():
    var_0 = 'replay_dir: /custom/replay'
    var_1 = 'cookiecutters_dir'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 9/25 statements.


import codecs as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = lambda x: x
    var_7 = 'utf-8'
    var_8 = module_0.open(var_0, encoding=var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_predicate_line_14_evaluates_to_true. Retrieved 18/28 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 14 (isinstance(yaml_dict, dict)) evaluates to True.'
    var_1 = 'config.yaml'
    var_2 = 'key: value\n'
    var_3 = 'utf-8'
    var_4 = 'os.path.exists'
    var_5 = True
    var_6 = 'builtins.open'
    var_7 = 'yaml.safe_load'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'merge_configs'
    var_12 = 'replay_dir'
    var_13 = 'cookiecutters_dir'
    var_14 = '/tmp'
    var_15 = {var_8: var_9, var_12: var_14, var_13: var_14}
    var_16 = '_expand_path'
    var_17 = lambda x: x



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 5/14 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 1/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_path_expansion. Retrieved 1/8 statements.
# Partially parsed test_get_config_merge_with_defaults. Retrieved 5/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/tmp/replays'
    var_3 = '/tmp/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = 'replay_dir'
    var_7 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True

def test_case_0():
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies'
    var_1 = '~'
    var_2 = '~'

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replays'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_config_with_valid_yaml_file. Retrieved 2/7 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_empty_yaml_file. Retrieved 2/7 statements.
# Partially parsed test_get_config_with_environment_variables. Retrieved 4/9 statements.
# Partially parsed test_get_config_with_user_home_expansion. Retrieved 2/6 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 2/6 statements.
# Partially parsed test_get_config_with_nested_dict. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/home/user/replay'
    var_2 = 'config.yaml'
    var_3 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replay\ncookiecutters_dir: ~/cookies\n'
    var_2 = '~'
    var_3 = '~'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  key: value\n'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_line_14_predicate_evaluates_to_false. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'key: value\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/tmp'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = lambda x: x
    var_8 = bool(var_0)
    assert var_8 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_evaluates_to_empty_dict. Retrieved 12/24 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp/replays'
    var_5 = '/tmp/cookies'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = lambda x: x
    var_8 = None
    var_9 = module_0.safe_load(var_8)
    var_10 = {}
    var_11 = var_9 or var_10
    var_12 = bool(var_11 == {})
    assert var_12 is True
    var_13 = bool(var_2)
    assert var_13 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_empty_dict. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-empty dict.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = var_0 or var_4
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = len(var_5)
    var_8 = bool(var_7 > 0)
    assert var_8 is True
    var_9 = bool(var_5)
    assert var_9 is True
    var_10 = var_5 or {}
    var_11 = bool((var_5 or {}) == var_5)
    assert var_11 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unable to parse YAML file'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_defaults_to_empty_dict. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_yaml_error_predicate_evaluates_to_false. Retrieved 3/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unable to parse YAML file'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 3/10 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/10 statements.
# Partially parsed test_get_config_valid_yaml_with_path_expansion. Retrieved 1/10 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/10 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 1/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = True
    assert var_3 is True

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = None
    var_2 = True
    assert var_2 is True

def test_case_0():
    var_0 = '- item1\n- item2\n'
    var_1 = None
    var_2 = True
    assert var_2 is True

def test_case_0():
    var_0 = 'replay_dir: ~/test_replay\ncookiecutters_dir: $HOME/test_cookies\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '~'
    var_5 = '$HOME'

def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True

def test_case_0():
    var_0 = 'replay_dir: /custom/replay\n'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_environment_variables. Retrieved 7/13 statements.
# Partially parsed test_get_config_with_user_home_expansion. Retrieved 6/16 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 4/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/home/user/replays'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/home/user/cookies'
    var_4 = 'config.yaml'
    var_5 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n'
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'
    var_5 = 'replay_dir'
    var_6 = '~'
    var_7 = 'cookiecutters_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_none_value. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-None dict.'
    var_1 = 'key: value\n'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = bool(var_4 == {'key': 'value'})
    assert var_5 is True
    var_6 = bool(var_4 is not {})
    assert var_6 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_yaml_error_not_raised. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'key: value\n'
    var_1 = bool(var_0)
    assert var_1 is True



