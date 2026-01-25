####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 1/3 statements.
# Partially parsed test_get_user_config_with_default_user_config. Retrieved 1/2 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/default/replay'
    var_3 = '/default/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/replay'
    var_3 = '/custom/cookiecutters'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = {var_0: var_2, var_1: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_with_valid_file. Retrieved 5/10 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters'
    var_2 = module_0.get_config(var_0)
    var_3 = '~/replays'
    var_4 = '~/cookiecutters'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_default_user_config. Retrieved 1/4 statements.


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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config.yaml'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = module_0.get_user_config()
    var_3 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_file_exists_returns_config_from_user_config_path. Retrieved 3/4 statements.
# Partially parsed test_config_file_does_not_exist_returns_default_config. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/user/config'
    var_1 = False
    var_2 = module_0.get_user_config()



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_default_config_dict. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_user_config_path. Retrieved 1/5 statements.
# Partially parsed test_get_user_config_user_config_not_found. Retrieved 1/2 statements.


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
    var_0 = '/path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config.yaml'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #10
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = module_0.get_config(var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_env_config_file. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_default_config_fallback. Retrieved 5/8 statements.


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
    var_0 = '/path/to/custom/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env/config.yaml'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = module_0.get_user_config()
    var_3 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'exists'
    var_3 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'exists'
    var_3 = lambda path: var_1
    var_4 = module_0.get_user_config()



