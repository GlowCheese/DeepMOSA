####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 1/2 statements.
# Partially parsed test_get_user_config_user_config_path_not_exists. Retrieved 5/7 statements.


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
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda path: var_1
    var_4 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_default_config_is_true. Retrieved 2/3 statements.
# Partially parsed test_default_config_is_dict. Retrieved 4/5 statements.
# Partially parsed test_env_config_file_is_not_set_and_user_config_path_does_not_exist. Retrieved 4/9 statements.
# Partially parsed test_env_config_file_is_set. Retrieved 3/5 statements.


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
    var_0 = '/custom/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = '/default/path/config.yaml'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/env/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/6 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_expand_path. Retrieved 5/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = var_2['replay_dir']
    assert var_3 == '/path/to/replay'
    var_4 = var_2['cookiecutters_dir']
    assert var_4 == '/path/to/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: {'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'expand_path_config.yaml'
    var_1 = 'replay_dir: $HOME/replay\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '$HOME/replay'
    var_4 = var_2['replay_dir']
    var_5 = '~/cookiecutters'
    var_6 = var_2['cookiecutters_dir']



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_returns_dict. Retrieved 2/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_config_with_valid_file. Retrieved 8/11 statements.
# Partially parsed test_get_config_with_invalid_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_top_level. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/expanded/replay'
    var_4 = '/expanded/cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'replay_dir: $HOME/replay\ncookiecutters_dir: ~/cookiecutters'
    var_7 = module_0.get_config(var_0)
    var_8 = var_7['replay_dir']
    var_9 = bool(var_7['replay_dir'] == var_5['replay_dir'])
    assert var_9 is True
    var_10 = var_7['cookiecutters_dir']
    var_11 = bool(var_7['cookiecutters_dir'] == var_5['cookiecutters_dir'])
    assert var_11 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid yaml'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/13 statements.
# Partially parsed test_get_user_config_with_env_var_config_file. Retrieved 8/15 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 8/15 statements.
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
    var_3 = '/custom/cookies'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/custom_replay'
    var_4 = '~/custom_cookies'
    var_5 = {var_1: var_3, var_2: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/env_replay'
    var_4 = '~/env_cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'COOKIECUTTER_CONFIG'
    var_7 = module_0.get_user_config()
    var_8 = var_7['replay_dir']
    var_9 = var_7['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutterrc'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/default_replay'
    var_4 = '~/default_cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_7 = module_0.get_user_config()
    var_8 = var_7['replay_dir']
    var_9 = var_7['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_3 = '/nonexistent/path'
    var_4 = module_0.get_user_config()



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = '/default/path/config.json'
    var_1 = bool(not (var_0 and var_0 is not var_0))
    assert var_1 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'not_a_dict'
    var_1 = 'test_config.yaml'
    var_2 = module_0.get_config(var_1)



# Parsed testcases at query #13
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_structure.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_file. Retrieved 1/4 statements.


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
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/5 statements.
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
    var_0 = '/env/config.yaml'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = module_0.get_user_config()
    var_3 = module_0.get_config(var_0)
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_yaml_safe_load_does_not_raise_exception. Retrieved 4/5 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.safe_load(var_0)
    var_2 = {}
    var_3 = var_1 or var_2



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_config_file_is_user_config_path.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_expand_path_with_env_var. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_user_home. Retrieved 2/3 statements.
# Partially parsed test_expand_path_with_both_env_var_and_user_home. Retrieved 2/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/dir'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/home/user/dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/dir'
    var_1 = module_0._expand_path(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/~/dir'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/home/user/~/dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/some/path'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_user_config_path. Retrieved 1/4 statements.


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
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = '/tmp/env_config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(var_0 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_config_file_is_valid_yaml_dict. Retrieved 2/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_config_valid_file. Retrieved 5/12 statements.
# Partially parsed test_get_config_invalid_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yml'
    var_1 = 'replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True
    var_8 = '$HOME/test'
    var_9 = var_2['replay_dir']
    var_10 = '~/cookies'
    var_11 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yml'
    var_1 = 'replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies\ninvalid_key: [1, 2, 3]'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_config.yml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yml'
    var_1 = 'replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies\ninvalid_yaml: - test'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yml'
    var_1 = '- replay_dir: $HOME/test\n- cookiecutters_dir: ~/cookies'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/8 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/6 statements.
# Partially parsed test_get_user_config_with_env_var_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 6/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/custom_replays'
    var_3 = '~/custom_templates'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    var_7 = var_5['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/path\ncookiecutters_dir: /templates'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/path\ncookiecutters_dir: /env/templates'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/path'
    var_5 = var_3['cookiecutters_dir']
    assert var_5 == '/env/templates'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutter.yaml'
    var_1 = 'replay_dir: /user/path\ncookiecutters_dir: /user/templates'
    var_2 = 'os.path.expanduser'
    var_3 = '~'
    var_4 = 1
    var_5 = module_0.get_user_config()
    var_6 = var_5['replay_dir']
    assert var_6 == '/user/path'
    var_7 = var_5['cookiecutters_dir']
    assert var_7 == '/user/templates'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #23
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_config_file_is_user_config_path.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'replay_dir: $HOME/test_replay\ncookiecutters_dir: ~/test_cookies'
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
    var_0 = 'nonexistent.yml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'merge_config.yml'
    var_1 = 'replay_dir: $HOME/test\ncookiecutters_dir: ~/test'
    var_2 = module_0.get_config(var_0)
    var_3 = 'abbreviations'
    var_4 = bool('abbreviations' in var_2)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 6/12 statements.


import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.open(var_0)
    var_3 = module_1.safe_load(var_2)
    var_4 = {}
    var_5 = var_3 or var_4



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_true. Retrieved 5/7 statements.


import codecs as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'w'
    var_2 = module_0.open(var_0, var_1)
    var_3 = 'key: value'
    var_4 = module_1.get_config(var_0)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_yaml_safe_load_returns_empty_dict_when_none. Retrieved 3/16 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = var_0 or var_1
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 5/12 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 5/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir: /test/replay\ncookiecutters_dir: /test/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '/test/replay'
    var_4 = var_2['replay_dir']
    var_5 = '/test/cookiecutters'
    var_6 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_config.yaml'
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
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = module_0.get_config(var_0)
    var_3 = 'replay_dir'
    var_4 = var_2['replay_dir']
    var_5 = 'cookiecutters_dir'
    var_6 = var_2['cookiecutters_dir']



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_with_valid_file. Retrieved 5/10 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_defaults. Retrieved 4/9 statements.
# Partially parsed test_get_config_expands_env_vars. Retrieved 4/9 statements.


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
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'partial_config.yaml'
    var_1 = 'replay_dir: ~/replays'
    var_2 = module_0.get_config(var_0)
    var_3 = 'cookiecutters_dir'
    var_4 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_var_config.yaml'
    var_1 = 'replay_dir: $TEST_ENV_VAR/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '$TEST_ENV_VAR/replays'
    var_4 = var_2['replay_dir']



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 5/11 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_empty_yaml. Retrieved 5/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True
    var_8 = '~/replays'
    var_9 = var_2['replay_dir']
    var_10 = '~/cookiecutters'
    var_11 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True
    var_4 = 'replay_dir'
    var_5 = bool('replay_dir' in var_2)
    assert var_5 is True
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_2)
    assert var_7 is True
    var_8 = 'replay_dir'
    var_9 = var_2['replay_dir']
    var_10 = 'cookiecutters_dir'
    var_11 = var_2['cookiecutters_dir']



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_config_returns_dict_when_valid_yaml_file_is_provided. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key: value'
    var_1 = 'valid_config.yaml'
    var_2 = module_0.get_config(var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_get_config_returns_merged_config. Retrieved 7/18 statements.
# Partially parsed test_get_config_raises_exception_when_file_not_exists. Retrieved 1/6 statements.
# Partially parsed test_get_config_raises_exception_when_invalid_yaml. Retrieved 1/10 statements.
# Partially parsed test_get_config_raises_exception_when_top_level_not_dict. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/test_replays'
    var_3 = '$HOME/test_cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'replay_dir: ~/test_replays\n'
    var_6 = 'cookiecutters_dir: $HOME/test_cookiecutters\n'

def test_case_0():
    var_0 = '/nonexistent/path/to/config.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'invalid: yaml: here: {'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '- item1\n- item2\n'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_config_path_exists_and_readable. Retrieved 1/14 statements.


def test_case_0():
    var_0 = b'key: value'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_invalid_yaml_file. Retrieved 2/8 statements.
# Partially parsed test_non_dict_yaml_file. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '/tmp/invalid.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = '/tmp/non_dict.yaml'
    var_1 = '- item1\n- item2'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_yaml_dict_not_dict. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'not a dict'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 1/13 statements.


def test_case_0():
    var_0 = b'key: value'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_config_parser_rejects_non_dict_yaml. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_non_dict.yaml'
    var_1 = '- item1\n- item2'



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_config_file_parsed_successfully. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1)
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/6 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 9/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: $HOME/cookiecutters'
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
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

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
    var_1 = 'replay_dir: /custom/replays'
    var_2 = module_0.get_config(var_0)
    var_3 = var_2['replay_dir']
    assert var_3 == '/custom/replays'
    var_4 = 'cookiecutters_dir'
    var_5 = bool('cookiecutters_dir' in var_2)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path_config.yaml'
    var_1 = 'replay_dir: ~/test\ncookiecutters_dir: $HOME/test'
    var_2 = module_0.get_config(var_0)
    var_3 = 'replay_dir'
    var_4 = var_2[var_3]
    var_5 = '~'
    var_6 = 'cookiecutters_dir'
    var_7 = var_2[var_6]
    var_8 = '$HOME'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not a dictionary'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 4/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_content'
    var_1 = 'dummy_path'
    var_2 = module_0.get_config(var_1)
    var_3 = {}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/14 statements.
# Partially parsed test_get_user_config_with_env_var_config_file. Retrieved 8/16 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 7/17 statements.


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

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/custom_replay'
    var_4 = '~/custom_cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '~/env_replay'
    var_4 = '~/env_cookiecutters'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'COOKIECUTTER_CONFIG'
    var_7 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/default_replay'
    var_3 = '~/default_cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'os.path.exists'
    var_6 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



