####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 7/12 statements.
# Partially parsed test_get_user_config_default_fallback. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_invalid_config_file. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookiecutters'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookiecutters'
    var_2 = 'builtins.__import__'
    var_3 = lambda *args, **kwargs: __import__(*args, **kwargs)
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = ''
    var_6 = module_0.get_user_config()

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
    var_1 = 'invalid: [yaml content'
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 1/8 statements.
# Partially parsed test_get_config_empty_file. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_valid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_env_variables. Retrieved 2/12 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 4/17 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 1/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = '- item1\n- item2\n'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test: value'
    var_2 = 'test'
    var_3 = 'value'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_expand_path_with_home_directory. Retrieved 3/6 statements.
# Partially parsed test_expand_path_with_environment_variable. Retrieved 3/5 statements.
# Partially parsed test_expand_path_with_both_home_and_env_var. Retrieved 2/4 statements.
# Partially parsed test_expand_path_with_multiple_env_vars. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test_file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~/test_file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '/test/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/$HOME_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/to/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path/to/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'relative/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == 'relative/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._expand_path(var_0)
    assert var_1 == ''

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$VAR1/$VAR2/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = 'first/second/file.txt'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_path_does_not_exist. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True (file does not exist).'
    var_1 = 'non_existent_config.yaml'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_nonexistent_env_variable. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 6/15 statements.
# Partially parsed test_get_user_config_default_config_false_no_env_no_user_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_config_file_different_from_user_config_path. Retrieved 5/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'user_config.yaml'
    var_3 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_4 = 'os.path.exists'
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config(default_config=var_1)

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/path\ncookiecutters_dir: /custom/cookies'
    var_2 = 'os.path.exists'
    var_3 = True
    var_4 = lambda x: var_3



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_raises_when_config_path_does_not_exist. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Test that get_config raises ConfigDoesNotExistException when config file does not exist.'
    var_1 = 'non_existent_config.yaml'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 11/19 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = '/default/config/path'
    var_4 = 'default'
    var_5 = 'config'
    var_6 = {var_4: var_5}
    var_7 = None
    var_8 = False
    var_9 = module_0.get_user_config(var_7, var_8)
    var_10 = module_0.get_user_config(var_3, var_8)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_loads_user_config_when_file_exists. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"test": "value"}'
    var_2 = 'default'
    var_3 = 'config'
    var_4 = {var_2: var_3}
    var_5 = 'loaded'
    var_6 = {var_5: var_3}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_user_config_path_exists_predicate. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 'test'
    var_2 = 'config'
    var_3 = {var_1: var_2}
    var_4 = 'default'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 13/18 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'USER_CONFIG_PATH'
    var_3 = var_0 is not var_2
    var_4 = var_0 and var_3
    assert var_4 is False
    var_5 = ''
    var_6 = False
    var_7 = var_5 is not var_2
    var_8 = var_5 and var_7
    assert var_8 is False
    var_9 = '/home/user/.cookiecutter'
    var_10 = var_9
    var_11 = var_10 is not var_9
    var_12 = var_10 and var_11
    assert var_12 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_config_predicate_line_3_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True when config file exists.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_config_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_env_variable_valid. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_with_env_variable_invalid. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_default_when_no_env_and_no_user_config. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 8/17 statements.
# Partially parsed test_get_user_config_with_user_config_path_and_default_config_false. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = '/nonexistent/env/config.yaml'
    var_2 = module_0.get_user_config()

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
    var_1 = 'replay_dir: /user/replay'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = True
    var_6 = 'builtins.open'
    var_7 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/path'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = None
    var_3 = module_0.get_user_config(var_2, var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = False
    var_6 = True
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_yaml_not_dict. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 6/15 statements.
# Partially parsed test_get_user_config_no_env_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_expandvars. Retrieved 4/10 statements.
# Partially parsed test_get_user_config_default_config_false_no_file. Retrieved 6/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = 'user_config'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

def test_case_0():
    var_0 = 'config_with_vars.yaml'
    var_1 = 'replay_dir: $HOME/replay\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'HOME'
    var_3 = '/home/testuser'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = None
    var_5 = module_0.get_user_config(var_4, var_1)



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = '/home/user/.cookiecutter_config'
    var_1 = var_0
    var_2 = var_1 is not var_0
    var_3 = var_1 and var_2
    assert var_3 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 10/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = '/default/path'
    var_4 = None
    var_5 = False
    var_6 = module_0.get_user_config(var_4, var_5)
    var_7 = module_0.get_user_config(var_3, var_5)
    var_8 = ''
    var_9 = module_0.get_user_config(var_8, var_5)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_config_raises_when_config_path_does_not_exist. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Test that get_config raises ConfigDoesNotExistException when config file does not exist.'
    var_1 = 'non_existent_config.yaml'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = var_3[var_4]
    var_6 = False
    var_7 = True
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 3/17 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = module_0.get_user_config()



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = '/home/user/.cookiecutter_config.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = var_0
    var_5 = var_4 is not var_0
    var_6 = var_4 and var_5
    assert var_6 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_not_dict. Retrieved 1/8 statements.
# Partially parsed test_get_config_empty_file. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_valid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 2/12 statements.
# Partially parsed test_get_config_with_user_home. Retrieved 3/13 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 1/8 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 1/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = '- item1\n- item2\n'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'
    var_1 = 'TEST_REPLAY_DIR'

def test_case_0():
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_1 = 'replay_dir'
    var_2 = '~'

def test_case_0():
    var_0 = 'replay_dir: /custom/replays\n'

def test_case_0():
    var_0 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  custom_key: custom_value\n'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 3/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = module_0.get_user_config()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_config_predicate_line_3_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True when config file exists.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 3/11 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/10 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/12 statements.
# Partially parsed test_get_user_config_returns_default_when_no_config_exists. Retrieved 5/8 statements.
# Partially parsed test_expand_path_with_env_variable. Retrieved 4/5 statements.
# Partially parsed test_expand_path_with_home. Retrieved 3/4 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_not_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_valid_yaml. Retrieved 4/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /test/replay\n'
    var_2 = False

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /custom/path\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/path\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'c'
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_5: var_7}
    var_9 = module_0.merge_configs(var_4, var_8)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 'c'
    var_10 = 20
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = {var_0: var_11, var_9: var_12}
    var_14 = module_0.merge_configs(var_8, var_13)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = 10
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.merge_configs(var_8, var_12)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'TEST_VAR'
    var_1 = '/test/path'
    var_2 = '$TEST_VAR/subdir'
    var_3 = module_0._expand_path(var_2)
    assert var_3 == '/test/path/subdir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test'
    var_1 = module_0._expand_path(var_0)
    var_2 = '/test'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /replay\ncookiecutters_dir: /cookies\n'
    var_2 = 'HOME'
    var_3 = '/home/user'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 5/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_line_40_predicate_evaluates_to_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = True
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 1/11 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/10 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 3/17 statements.
# Partially parsed test_get_config_expands_home_directory. Retrieved 3/14 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = '- item1\n- item2\n'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n'
    var_1 = 'TEST_REPLAY_DIR'
    var_2 = 'TEST_COOKIES_DIR'

def test_case_0():
    var_0 = 'replay_dir: ~/replay\ncookiecutters_dir: ~/cookies\n'
    var_1 = 'replay_dir'
    var_2 = '~'

def test_case_0():
    var_0 = 'replay_dir: /tmp/custom\n'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = True
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_nonexistent_env_variable. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'HOME'
    var_3 = '/nonexistent/home'
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = None
    var_3 = var_0 is not var_2
    var_4 = var_0 and var_3
    assert var_4 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_default_when_no_config_exists. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_config_file_different_from_user_path. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var_takes_precedence. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'HOME'
    var_3 = '/nonexistent'
    var_4 = module_0.get_user_config()

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/path\ncookiecutters_dir: /custom/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/merged/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = True
    var_5 = False
    assert var_5 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 4/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_env_variables. Retrieved 5/10 statements.
# Partially parsed test_get_config_with_user_home. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = 'utf-8'
    var_3 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/home/user/replays'
    var_2 = 'config.yaml'
    var_3 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'
    var_4 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_default_when_no_config_exists. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 7/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

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
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'

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
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = '__main__.USER_CONFIG_PATH'
    var_6 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_expand_path_with_environment_variable. Retrieved 4/11 statements.
# Partially parsed test_expand_path_with_home_directory. Retrieved 4/7 statements.
# Partially parsed test_expand_path_with_both_env_and_home. Retrieved 6/15 statements.
# Partially parsed test_expand_path_with_only_home_symbol. Retrieved 2/4 statements.


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
    var_0 = 'TEST_DIR'
    var_1 = '~/$TEST_DIR/file.txt'
    var_2 = module_0._expand_path(var_1)
    var_3 = '~'
    var_4 = 'mydir/file.txt'
    var_5 = 'TEST_DIR'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'relative/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == 'relative/path/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._expand_path(var_0)
    assert var_1 == ''

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~'
    var_1 = module_0._expand_path(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_config_predicate_line_3_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True when config file exists.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test: value'



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = '/path/to/config'
    var_1 = '/path/to/config'
    var_2 = var_0 is not var_1
    var_3 = var_0 and var_2
    assert var_3 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_config_predicate_line_3_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 3 evaluates to True when config file exists.'
    var_1 = 'config.yaml'
    var_2 = 'key: value\n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file_and_default_config_dict. Retrieved 5/10 statements.
# Partially parsed test_get_user_config_with_invalid_yaml_file. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom/replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

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
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookiecutters'

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookiecutters'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content:'
    var_2 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'new_key'
    var_2 = '/custom/path'
    var_3 = 'new_value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = module_0.get_user_config(default_config=var_0)



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = '/path/to/config'
    var_1 = '/path/to/config'
    var_2 = var_0 is not var_1
    var_3 = var_0 and var_2
    assert var_3 is False



