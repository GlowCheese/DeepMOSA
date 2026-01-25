####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_config_with_valid_yaml_file. Retrieved 5/12 statements.
# Partially parsed test_get_config_with_invalid_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_top_level_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_default_config. Retrieved 5/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: $HOME/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = 'replay_dir'
    var_4 = bool('replay_dir' in var_2)
    assert var_4 is True
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_2)
    assert var_6 is True
    var_7 = '$HOME/replays'
    var_8 = var_2['replay_dir']
    var_9 = '~/cookiecutters'
    var_10 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'merge_config.yaml'
    var_1 = 'replay_dir: $HOME/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '$HOME/replays'
    var_4 = var_2['replay_dir']
    var_5 = '~/cookiecutters'
    var_6 = var_2['cookiecutters_dir']



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

def test_case_0():
    var_0 = 'custom_config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/env/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutterrc'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_2 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/user/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/user/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 1/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'KeyError'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = True
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_config_file_exists. Retrieved 4/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = True
    var_2 = {}
    var_3 = module_0.get_config(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/path/to/existing/config'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_path_does_not_exist. Retrieved 1/9 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 5/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/env/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutter.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_2 = 'os.path.expanduser'
    var_3 = '~/.cookiecutter.yaml'
    var_4 = module_0.get_user_config()
    var_5 = var_4['replay_dir']
    assert var_5 == '/user/replay'
    var_6 = var_4['cookiecutters_dir']
    assert var_6 == '/user/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_expand_path_with_env_var. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_user_home. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_both_env_var_and_user_home. Retrieved 2/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/path'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/~/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/test/~/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/some/path'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_custom_config_file. Retrieved 1/6 statements.
# Partially parsed test_get_user_config_env_var_config. Retrieved 3/9 statements.
# Partially parsed test_get_user_config_user_config_exists. Retrieved 2/8 statements.


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

def test_case_0():
    var_0 = 'replay_dir: /custom/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir: /env/path'
    var_1 = module_0.get_user_config()
    var_2 = var_1['replay_dir']
    assert var_2 == '/env/path'
    var_3 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir: /user/path'
    var_1 = module_0.get_user_config()
    var_2 = var_1['replay_dir']
    assert var_2 == '/user/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_predicate_at_line_33_evaluates_to_false_when_config_file_is_user_config_path.


def test_case_0():
    var_0 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 5/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = None
    var_3 = module_0.get_user_config()
    var_4 = module_0.get_config(var_0)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_at_line_33_evaluates_to_false.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = module_0.get_user_config()



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = '/path/to/custom/config'
    var_1 = '/path/to/user/config'
    var_2 = bool(not (var_0 and var_0 is not var_1))
    assert var_2 is True

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = '/path/to/user/config'
    var_2 = bool(not (var_0 and var_0 is not var_1))
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_config_file. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'
    var_2 = 'module.USER_CONFIG_PATH'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/default/path/to/config'
    var_1 = module_0.get_user_config(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/path/to/existing/config'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = '/default/path'
    var_2 = bool(not (var_0 and var_0 is not var_1))
    assert var_2 is True
    var_3 = '/default/path'
    var_4 = bool(not (var_3 and var_3 is not var_1))
    assert var_4 is True
    var_5 = '/custom/path'
    var_6 = '/default/path'
    var_7 = bool(var_5 and var_5 is not var_6)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 12/18 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 15/22 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 11/17 statements.
# Partially parsed test_get_user_config_with_no_config_found. Retrieved 5/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/custom_replay'
    var_3 = '~/custom_cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'builtins.open'
    var_6 = 'replay_dir: ~/custom_replay\ncookiecutters_dir: ~/custom_cookiecutters'
    var_7 = 'yaml.safe_load'
    var_8 = 'os.path.exists'
    var_9 = True
    var_10 = '/custom/config.yaml'
    var_11 = module_0.get_user_config(var_10)
    var_12 = var_11['replay_dir']
    var_13 = var_11['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/env_replay'
    var_3 = '~/env_cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'os.environ'
    var_6 = 'COOKIECUTTER_CONFIG'
    var_7 = '/env/config.yaml'
    var_8 = {var_6: var_7}
    var_9 = 'builtins.open'
    var_10 = 'replay_dir: ~/env_replay\ncookiecutters_dir: ~/env_cookiecutters'
    var_11 = 'yaml.safe_load'
    var_12 = 'os.path.exists'
    var_13 = True
    var_14 = module_0.get_user_config()
    var_15 = var_14['replay_dir']
    var_16 = var_14['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/default_replay'
    var_3 = '~/default_cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'builtins.open'
    var_6 = 'replay_dir: ~/default_replay\ncookiecutters_dir: ~/default_cookiecutters'
    var_7 = 'yaml.safe_load'
    var_8 = 'os.path.exists'
    var_9 = True
    var_10 = module_0.get_user_config()
    var_11 = var_10['replay_dir']
    var_12 = var_10['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'os.environ'
    var_1 = True
    var_2 = 'os.path.exists'
    var_3 = False
    var_4 = module_0.get_user_config()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 5/10 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 5/12 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '~/replays'
    var_4 = var_2['replay_dir']
    var_5 = '~/cookiecutters'
    var_6 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: ['
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = module_0.get_config(var_0)
    var_3 = 'replay_dir'
    var_4 = var_2['replay_dir']
    var_5 = 'cookiecutters_dir'
    var_6 = var_2['cookiecutters_dir']



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_var_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 1/2 statements.
# Partially parsed test_get_user_config_user_config_path_not_exists. Retrieved 5/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config.yaml'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = module_0.get_user_config()
    var_3 = module_0.get_config(var_0)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_predicate_at_line_33_evaluates_to_false.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_var. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_default_user_config_path. Retrieved 1/3 statements.
# Failed to parse test_get_user_config_user_config_path.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/env/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True
    var_4 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_config_file_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 6/12 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom/replay'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'
    var_2 = 'os.path.exists'
    var_3 = 'os.environ'
    var_4 = {}
    var_5 = module_0.get_user_config()
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_True. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_with_existing_file. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = 'replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = var_2['replay_dir']
    assert var_4 == '/path/to/replay'
    var_5 = var_2['cookiecutters_dir']
    assert var_5 == '/path/to/cookiecutters'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 1/2 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_user_config_path. Retrieved 1/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/env/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True
    var_4 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_expand_path_with_home_dir. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_env_var. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_both_home_and_env. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~'
    var_1 = module_0._expand_path(var_0)
    var_2 = bool(var_1 != var_0)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_PATH/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/tmp/file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/$TEST_DIR/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'
    var_3 = 'documents/file.txt'
    var_4 = bool('documents/file.txt' in var_1)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = bool(var_1 == var_0)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$UNKNOWN_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/file.txt'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = '/path/to/user/config'
    var_2 = bool(not (var_0 and var_0 is not var_1))
    assert var_2 is True
    var_3 = '/path/to/user/config'
    var_4 = bool(not (var_3 and var_3 is not var_1))
    assert var_4 is True
    var_5 = '/different/path'
    var_6 = bool(var_5 and var_5 is not var_1)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 8/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'replay_dir: ~/test_replay\ncookiecutters_dir: ~/test_cookies\n'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent.yml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'replay_dir: ~/test_replay\ncookiecutters_dir: ~/test_cookies\n'
    var_2 = module_0.get_config(var_0)
    var_3 = 'replay_dir'
    var_4 = var_2[var_3]
    var_5 = '~'
    var_6 = 'cookiecutters_dir'
    var_7 = var_2[var_6]



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_file_exists. Retrieved 11/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replays'
    var_5 = 'cookies'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = lambda x, y: y
    var_9 = lambda x: x
    var_10 = module_0.get_config(var_0)
    var_11 = bool(var_10 == {'replay_dir': 'replays', 'cookiecutters_dir': 'cookies'})
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_no_config_found. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/custom/config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/replay'
    var_4 = '/custom/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config(var_0)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/env/replay'
    var_4 = '/env/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config()
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/default/replay'
    var_3 = '/default/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config()
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists. Retrieved 2/7 statements.
# Partially parsed test_predicate_at_line_43_evaluates_to_false_when_user_config_path_does_not_exist. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/tmp/test_config'
    var_1 = 'test'
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = '/tmp/nonexistent_config'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 7/8 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 8/12 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 6/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/replay'
    var_4 = '/custom/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/env/config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/env/replay'
    var_4 = '/env/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config()
    var_7 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/user/replay'
    var_3 = '/user/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/invalid/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/invalid/yaml.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non_dict/yaml.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_config_file_exists. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'existing_config.yml'
    var_1 = True
    var_2 = module_0.get_config(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 1/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/env/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True
    var_4 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = None
    var_3 = module_0.get_user_config()
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_default_config_file. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_invalid_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_non_dict_yaml. Retrieved 2/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookiecutters'
    var_2 = module_0.get_user_config()
    var_3 = var_2['replay_dir']
    assert var_3 == '/custom/replay'
    var_4 = var_2['cookiecutters_dir']
    assert var_4 == '/custom/cookiecutters'
    var_5 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yml'
    var_1 = module_0.get_user_config(var_0)

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'invalid: yaml: file'

def test_case_0():
    var_0 = 'config.yml'
    var_1 = '- item1\n- item2'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_config_with_valid_file. Retrieved 2/3 statements.
# Partially parsed test_get_config_with_expanded_paths. Retrieved 6/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = 'replay_dir'
    var_3 = bool('replay_dir' in var_1)
    assert var_3 is True
    var_4 = 'cookiecutters_dir'
    var_5 = bool('cookiecutters_dir' in var_1)
    assert var_5 is True

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
    var_0 = 'non_dict_top_level_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = 'replay_dir'
    var_3 = var_1[var_2]
    var_4 = var_1['replay_dir']
    var_5 = 'cookiecutters_dir'
    var_6 = var_1[var_5]
    var_7 = var_1['cookiecutters_dir']



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 1/2 statements.
# Partially parsed test_get_user_config_user_config_path_not_exists. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/config'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_user_config_exists. Retrieved 4/8 statements.


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

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir: /custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yml'
    var_1 = 'replay_dir: /env/replay'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutterrc'
    var_1 = 'replay_dir: /user/replay'
    var_2 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_config_file_exists. Retrieved 1/9 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/6 statements.
# Partially parsed test_get_user_config_custom_config_file. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 1/2 statements.
# Partially parsed test_get_user_config_user_config_path_not_exists. Retrieved 2/5 statements.


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
    var_0 = '/tmp/custom_config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 5/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = True
    var_2 = False
    var_3 = module_0.get_user_config()
    var_4 = module_0.get_config(var_0)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/env/config.yaml'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = module_0.get_user_config()
    var_3 = module_0.get_config(var_0)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay'
    var_2 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = '/some/path'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_config_file_exists. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: "~/replays"\ncookiecutters_dir: "~/cookiecutters"'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'partial_config.yaml'
    var_1 = 'replay_dir: "~/custom_replays"'
    var_2 = module_0.get_config(var_0)
    var_3 = '~/custom_replays'
    var_4 = var_2['replay_dir']
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_2)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_default_user_config. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/path/to/env/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True
    var_4 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/user/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



