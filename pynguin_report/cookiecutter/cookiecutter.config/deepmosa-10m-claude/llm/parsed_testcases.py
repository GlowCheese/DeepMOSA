####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_no_args_no_env_no_user_config. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 9/16 statements.
# Partially parsed test_get_user_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_yaml_not_dict. Retrieved 3/7 statements.
# Partially parsed test_expand_path_with_home. Retrieved 3/4 statements.
# Partially parsed test_expand_path_with_env_var. Retrieved 4/5 statements.


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

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/env/replay'

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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list_config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

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
    var_10 = var_9['a']
    assert var_10 == 1
    var_11 = var_9['b']
    assert var_11 == 3
    var_12 = var_9['c']
    assert var_12 == 4

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
    var_15 = var_14['a']['x']
    assert var_15 == 1
    var_16 = var_14['a']['y']
    assert var_16 == 20
    var_17 = var_14['b']
    assert var_17 == 3
    var_18 = var_14['c']
    assert var_18 == 4

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
    var_9 = 20
    var_10 = {var_3: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.merge_configs(var_8, var_12)
    var_14 = var_13['a']['b']['c']
    assert var_14 == 1
    var_15 = var_13['a']['b']['d']
    assert var_15 == 20

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test/path'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'
    var_3 = bool('~' not in var_1)
    assert var_3 is True
    var_4 = '/'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'TEST_VAR'
    var_1 = '/test/value'
    var_2 = '$TEST_VAR/path'
    var_3 = module_0._expand_path(var_2)
    assert var_3 == '/test/value/path'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_line_33_evaluates_to_false. Retrieved 7/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = False
    var_5 = bool(var_2 is not None)
    assert var_5 is True
    var_6 = ''
    var_7 = False
    var_8 = module_0.get_user_config(var_6, var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = '/path/to/default/config'
    var_1 = '/path/to/default/config'
    var_2 = var_0 is not var_1
    var_3 = var_0 and var_2
    assert var_3 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/4 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_default_path_exists. Retrieved 8/17 statements.
# Partially parsed test_get_user_config_default_path_not_exists. Retrieved 6/11 statements.
# Partially parsed test_get_user_config_invalid_config_file. Retrieved 2/5 statements.
# Partially parsed test_get_user_config_with_malformed_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_priority_default_config_dict_over_file. Retrieved 3/4 statements.


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
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /default/replay\ncookiecutters_dir: /default/cookies'
    var_2 = 'os.path.exists'
    var_3 = 'builtins.open'
    var_4 = 1
    var_5 = 'COOKIECUTTER_CONFIG'
    var_6 = False
    var_7 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = ''
    var_2 = 'os.path.exists'
    var_3 = False
    var_4 = lambda path: var_3
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'malformed_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/priority/replay'
    var_2 = {var_0: var_1}

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_valid_config_with_paths. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_home_directory. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/6 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content:'
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '- item1\n- item2'
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
    var_1 = 'replay_dir: /test/replay\ncookiecutters_dir: /test/cookies'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replay\ncookiecutters_dir: ~/cookies'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'some_nested_config:\n  key1: value1'
    var_2 = 'utf-8'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_line_40_predicate_evaluates_to_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 7/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = True
    var_6 = False
    assert var_6 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_line_43_evaluates_to_true. Retrieved 7/18 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test: value'
    var_2 = 'test'
    var_3 = 'value'
    var_4 = None
    var_5 = False
    var_6 = module_0.get_user_config(var_4, var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_config_predicate_line_3_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_user_config_path_exists. Retrieved 4/19 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'config'
    var_2 = 'default'
    var_3 = module_0.get_user_config()



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = '/default/path'
    var_2 = var_0 is not var_1
    var_3 = var_0 and var_2
    assert var_3 is False
    var_4 = '/default/path'
    var_5 = '/default/path'
    var_6 = var_4 is not var_5
    var_7 = var_4 and var_6
    assert var_7 is False
    var_8 = ''
    var_9 = '/default/path'
    var_10 = var_8 is not var_9
    var_11 = var_8 and var_10
    assert var_11 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 8/15 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default'
    var_2 = None
    var_3 = False
    var_4 = module_0.get_user_config(var_2, var_3)
    var_5 = bool(var_4 == {'key': 'default'})
    assert var_5 is True
    var_6 = '/home/user/.cookiecutterrc'
    var_7 = False
    var_8 = module_0.get_user_config(var_6, var_7)
    var_9 = bool(var_8 == {'key': 'default'})
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false_and_no_env_and_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_user_config_path_exists. Retrieved 8/17 statements.
# Partially parsed test_get_user_config_config_file_not_found. Retrieved 5/7 statements.
# Partially parsed test_get_user_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_yaml_not_dict. Retrieved 3/7 statements.
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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config(default_config=var_1)

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = bool('/env/replay' in var_3['replay_dir'] or var_3['replay_dir'] == '/env/replay')
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'user_config.yaml'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = True
    var_6 = 'builtins.open'
    var_7 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'os.path.exists'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = '/nonexistent/config.yaml'
    var_4 = module_0.get_user_config(var_3)
    var_5 = bool(False)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list_config.yaml'
    var_1 = '- item1\n- item2\n'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'partial_config.yaml'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'cookiecutters_dir'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_custom_config_file_not_equal_to_user_config_path. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_default_when_no_config_exists. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_from_environment_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_from_user_config_path. Retrieved 9/16 statements.
# Partially parsed test_get_user_config_priority_env_over_user_path. Retrieved 5/13 statements.
# Partially parsed test_get_user_config_false_default_config_parameter. Retrieved 5/8 statements.


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

def test_case_0():
    var_0 = 'custom_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

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
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = True
    var_4 = 'builtins.open'
    var_5 = 'io'
    var_6 = __import__(var_5)
    var_7 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_8 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'extra_key'
    var_2 = '/custom/replay'
    var_3 = 'extra_value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/replay'
    var_7 = 'extra_key'
    var_8 = bool('extra_key' in var_5)
    assert var_8 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config(default_config=var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_path_does_not_exist. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 1/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_valid_config. Retrieved 1/9 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 2/12 statements.
# Partially parsed test_get_config_with_user_home. Retrieved 3/13 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 1/8 statements.


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
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

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
    var_1 = '~'
    var_2 = '~'
    var_3 = 'replay_dir'
    var_4 = '~'

def test_case_0():
    var_0 = 'replay_dir: /custom/replays\n'
    var_1 = 'cookiecutters_dir'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_with_nonexistent_env_variable_and_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_default_config_false_over_custom_config_file. Retrieved 3/8 statements.
# Partially parsed test_get_user_config_default_config_dict_overrides_config_file. Retrieved 5/10 statements.
# Partially parsed test_get_user_config_invalid_yaml_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_yaml_not_dict. Retrieved 3/7 statements.


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
    var_2 = '/tmp/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = '/env/replay'
    var_5 = bool('/env/replay' in var_3['replay_dir'])
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'
    var_2 = False
    var_3 = '/custom/replay'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'
    var_2 = 'replay_dir'
    var_3 = '/override/replay'
    var_4 = {var_2: var_3}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'notdict.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_user_config(var_0)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_raises_exception_when_config_file_does_not_exist. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = True
    assert var_5 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_default_config_false_no_env_no_user_config. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 7/13 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 12/21 statements.
# Partially parsed test_get_user_config_config_file_not_equal_user_config_path. Retrieved 7/14 statements.
# Partially parsed test_get_user_config_invalid_yaml_file. Retrieved 3/7 statements.


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
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config(default_config=var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'os.path.exists'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = module_0.get_user_config(default_config=var_4)
    var_7 = var_6['replay_dir']
    assert var_7 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '.cookiecutterrc'
    var_1 = 'replay_dir: /user/replay\ncookiecutters_dir: /user/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = False
    var_4 = 'os.path.exists'
    var_5 = 'cookiecutter'
    var_6 = 'os.path.expandvars'
    var_7 = lambda x: x
    var_8 = 'os.path.expanduser'
    var_9 = lambda x: x
    var_10 = None
    var_11 = module_0.get_user_config(var_10, var_3)

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'
    var_2 = 'os.path.expandvars'
    var_3 = lambda x: x
    var_4 = 'os.path.expanduser'
    var_5 = lambda x: x
    var_6 = False

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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_file. Retrieved 3/7 statements.
# Partially parsed test_get_config_valid_config. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 5/9 statements.
# Partially parsed test_get_config_with_tilde_expansion. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 3/6 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content:'
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
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/custom/replays'
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
    var_1 = 'replay_dir: /custom/path\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  key1: value1\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_expand_path_with_home_directory. Retrieved 3/6 statements.
# Partially parsed test_expand_path_with_environment_variable. Retrieved 3/5 statements.
# Partially parsed test_expand_path_with_both_home_and_env_var. Retrieved 2/4 statements.
# Partially parsed test_expand_path_with_only_home_symbol. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/test_file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~/test_file.txt'
    var_3 = '~'
    var_4 = bool('~' not in var_1)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '/test/path/file.txt'
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/$HOME_TEST/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = '$HOME_TEST'
    var_3 = bool('$HOME_TEST' not in var_1)
    assert var_3 is True
    var_4 = '~'
    var_5 = bool('~' not in var_1)
    assert var_5 is True

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
    var_0 = '~'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 4/8 statements.
# Partially parsed test_get_config_empty_file. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/8 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 6/18 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/8 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/8 statements.


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
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: $HOME/replays\ncookiecutters_dir: $HOME/cookies\n'
    var_2 = 'utf-8'
    var_3 = '$HOME'
    var_4 = '$HOME'
    var_5 = 'replay_dir'
    var_6 = '~'
    var_7 = 'cookiecutters_dir'

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 6/12 statements.
# Partially parsed test_get_user_config_with_invalid_env_variable. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_no_env_no_user_config. Retrieved 5/8 statements.


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
    var_0 = '/nonexistent/path/config.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /env/replays\ncookiecutters_dir: /env/cookies'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = 'HOME'
    var_4 = False
    var_5 = module_0.get_user_config()
    var_6 = 'replay_dir'
    var_7 = bool('replay_dir' in var_5)
    assert var_7 is True

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
    var_1 = '/override/replay'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = module_0.get_user_config(var_3, var_2)
    var_5 = var_4['replay_dir']
    assert var_5 == '/override/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/8 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 5/10 statements.
# Partially parsed test_get_user_config_default_when_no_env_and_no_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_loads_user_config_when_exists. Retrieved 8/18 statements.
# Partially parsed test_get_user_config_with_invalid_config_file. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_nonexistent_config_file. Retrieved 2/5 statements.
# Partially parsed test_get_user_config_prioritizes_custom_file_over_default. Retrieved 3/8 statements.


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
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = module_0.get_user_config()
    var_5 = 'replay_dir'
    var_6 = bool('replay_dir' in var_4)
    assert var_6 is True

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
    var_1 = 'replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = 'os.path.exists'
    var_6 = '__main__.USER_CONFIG_PATH'
    var_7 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: : yaml: content:\n'
    var_2 = 'utf-8'
    var_3 = module_0.get_user_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'extra_key'
    var_2 = '/custom/path'
    var_3 = 'extra_value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/custom/path'
    var_7 = 'cookiecutters_dir'
    var_8 = bool('cookiecutters_dir' in var_5)
    assert var_8 is True

def test_case_0():
    var_0 = 'custom.yaml'
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies\n'
    var_2 = 'utf-8'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_opens_file_with_utf8_encoding. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_2 = 'utf-8'
    var_3 = 'utf-8'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_valid_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 2/6 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 2/6 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 4/8 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 2/5 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'

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
    var_1 = ''

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/test/replays'
    var_2 = 'config.yaml'
    var_3 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = '~'
    var_3 = '~'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'cookiecutters_dir'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 5/9 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 6/13 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/6 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content:'
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'non_dict.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'utf-8'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/home/user/replays'
    var_2 = 'config.yaml'
    var_3 = 'replay_dir: $TEST_REPLAY_DIR'
    var_4 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/.cookiecutters'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'
    var_5 = 'replay_dir'
    var_6 = '/'
    var_7 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'abbreviations:\n  custom_key: custom_value'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unable to parse YAML file'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 6/13 statements.
# Partially parsed test_get_user_config_user_config_path_exists. Retrieved 7/19 statements.
# Partially parsed test_get_user_config_returns_default_when_no_config_found. Retrieved 6/8 statements.
# Partially parsed test_get_user_config_with_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_expands_environment_variables. Retrieved 5/10 statements.
# Partially parsed test_get_user_config_expands_user_home. Retrieved 5/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns DEFAULT_CONFIG when default_config is True.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config merges provided dict with DEFAULT_CONFIG.'
    var_1 = 'replay_dir'
    var_2 = '/custom/replay'
    var_3 = {var_1: var_2}
    var_4 = module_0.get_user_config(default_config=var_3)
    var_5 = var_4['replay_dir']
    assert var_5 == '/custom/replay'
    var_6 = 'cookiecutters_dir'
    var_7 = bool('cookiecutters_dir' in var_4)
    assert var_7 is True

def test_case_0():
    var_0 = 'Test get_user_config loads custom config file.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /custom/path\ncookiecutters_dir: /cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config loads config from COOKIECUTTER_CONFIG environment variable.'
    var_1 = 'env_config.yaml'
    var_2 = 'replay_dir: /env/replay'
    var_3 = 'COOKIECUTTER_CONFIG'
    var_4 = False
    var_5 = module_0.get_user_config()
    var_6 = var_5['replay_dir']
    assert var_6 == '/env/replay'

def test_case_0():
    var_0 = 'Test get_user_config loads from USER_CONFIG_PATH when it exists.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = 'user_config.yaml'
    var_4 = 'replay_dir: /user/replay'
    var_5 = 'os.path.exists'
    var_6 = 'get_user_config'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config returns DEFAULT_CONFIG when no config file exists.'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = 'os.path.exists'
    var_4 = lambda p: var_2
    var_5 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config raises InvalidConfiguration for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content:'
    var_3 = module_0.get_user_config(var_0)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config raises ConfigDoesNotExistException for nonexistent custom file.'
    var_1 = '/nonexistent/config.yaml'
    var_2 = module_0.get_user_config(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test get_user_config expands environment variables in paths.'
    var_1 = 'TEST_REPLAY_DIR'
    var_2 = '/expanded/replay'
    var_3 = 'config.yaml'
    var_4 = 'replay_dir: $TEST_REPLAY_DIR'

def test_case_0():
    var_0 = 'Test get_user_config expands ~ to user home directory.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: ~/replay'
    var_3 = '~'
    var_4 = 'replay_dir'
    var_5 = '/'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/4 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_default_path_exists. Retrieved 9/19 statements.
# Partially parsed test_get_user_config_no_env_no_default_path. Retrieved 5/9 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 2/3 statements.


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
    var_1 = 'replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies'
    var_2 = 'replay_dir'

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
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = ''
    var_4 = False
    var_5 = 'os.path.exists'
    var_6 = 'builtins.open'
    var_7 = 1
    var_8 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = False
    var_2 = 'os.path.exists'
    var_3 = lambda x: var_1
    var_4 = module_0.get_user_config()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config(default_config=var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_false. Retrieved 12/24 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = 'key'
    var_5 = 'value'
    var_6 = '.cookiecutterrc'
    var_7 = False
    var_8 = bool(var_2 is not None)
    assert var_8 is True
    var_9 = 'default'
    var_10 = 'config'
    var_11 = ''
    var_12 = False
    var_13 = module_0.get_user_config(var_11, var_12)
    var_14 = bool(var_13 == {'default': 'config'})
    assert var_14 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/6 statements.
# Partially parsed test_get_config_expands_env_vars. Retrieved 7/12 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/6 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/7 statements.


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
    var_1 = 'replay_dir: /test/replay\ncookiecutters_dir: /test/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/test/replays'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/test/cookies'
    var_4 = 'config.yaml'
    var_5 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n'
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'abbreviations:\n  custom: value\nreplay_dir: /test\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_empty_dict. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-empty dict.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_3 = 'utf-8'
    var_4 = {}
    var_5 = var_0 or var_4
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = bool(var_5 != {})
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_config_predicate_line_14_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 1/9 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 1/9 statements.
# Partially parsed test_get_config_empty_file. Retrieved 1/10 statements.
# Partially parsed test_get_config_with_valid_yaml. Retrieved 4/19 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 2/13 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 1/10 statements.


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
    var_0 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = '~'
    var_6 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: ~/cookies\n'
    var_1 = 'TEST_REPLAY_DIR'

def test_case_0():
    var_0 = 'replay_dir: ~/custom_replays\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 3/7 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/7 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'non_existent_config.yaml'
    var_1 = bool(False)
    assert var_1 is True

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
    var_1 = 'replay_dir: $HOME/replays\ncookiecutters_dir: $HOME/cookies\n'
    var_2 = 'utf-8'
    var_3 = '$HOME'
    var_4 = '$HOME'

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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_defaults_to_empty_dict. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = {}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_valid_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_environment_variables. Retrieved 7/13 statements.
# Partially parsed test_get_config_with_tilde_expansion. Retrieved 3/7 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 4/8 statements.
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
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'REPLAY_PATH'
    var_1 = '/home/user/replays'
    var_2 = 'COOKIES_PATH'
    var_3 = '/home/user/cookies'
    var_4 = 'config.yaml'
    var_5 = 'replay_dir: $REPLAY_PATH\ncookiecutters_dir: $COOKIES_PATH\n'
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n'
    var_2 = 'utf-8'
    var_3 = '~'
    var_4 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content:\n  - broken'
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_cookiecutter_config_env_var_not_set. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = {var_0: var_1}
    var_3 = '/home/user/.cookiecutterrc'
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = 'COOKIECUTTER_CONFIG'
    var_6 = True
    var_7 = False
    assert var_7 is False



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0 and var_2
    assert var_3 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_empty_dict. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /some/path\ncookiecutters_dir: /another/path\n'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_line_40_predicate_evaluates_to_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = 'COOKIECUTTER_CONFIG'
    var_2 = False
    var_3 = True
    var_4 = 'COOKIECUTTER_CONFIG'
    var_5 = False
    var_6 = True
    assert var_6 is False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test: value'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_false. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 2/7 statements.
# Partially parsed test_get_user_config_with_env_variable. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_without_env_variable_with_user_config_exists. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_without_env_variable_without_user_config. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_user_config_with_non_dict_yaml_root. Retrieved 3/7 statements.


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
    var_0 = 'env_config.yaml'
    var_1 = 'replay_dir: /tmp/env_replay\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = module_0.get_user_config()
    var_4 = var_3['replay_dir']
    assert var_4 == '/tmp/env_replay'

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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/custom'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/custom'
    var_5 = 'cookiecutters_dir'
    var_6 = bool('cookiecutters_dir' in var_3)
    assert var_6 is True

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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_evaluates_to_empty_dict. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = {}
    var_3 = lambda x: x
    var_4 = None
    var_5 = {}
    var_6 = var_4 or var_5
    var_7 = bool(var_6 == {})
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'test_key: test_value'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_line_14_evaluates_to_false. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 14 evaluates to False when yaml_dict is a dict.'
    var_1 = 'config.yaml'
    var_2 = 'key: value\n'
    var_3 = 'utf-8'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_yaml_safe_load_returns_non_empty_dict. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-empty dict.'
    var_1 = 'config.yaml'
    var_2 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_3 = 'utf-8'
    var_4 = {}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_config_predicate_line_8_evaluates_to_false. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'non_existent.yaml'
    var_7 = True
    var_8 = False
    assert var_8 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_invalid_yaml. Retrieved 4/8 statements.
# Partially parsed test_get_config_non_dict_top_level. Retrieved 4/8 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_valid_config. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_env_vars. Retrieved 7/13 statements.
# Partially parsed test_get_config_with_home_expansion. Retrieved 6/14 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/7 statements.


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
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/test/replays'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/test/cookies'
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
    var_6 = '/'
    var_7 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_yaml_dict_is_dict. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'key: value\n'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/tmp'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_yaml_error_predicate_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unable to parse YAML file'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_yaml_error_predicate_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_line_14_predicate_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'key: value\n'



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

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/tmp'
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = lambda x: x
    var_7 = 'utf-8'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_config_with_valid_dict_yaml. Retrieved 8/16 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replays'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dump(var_5, **var_6)
    var_8 = 'HOME'
    var_9 = 'replay_dir'
    var_10 = 'cookiecutters_dir'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_get_config_opens_file_with_utf8_encoding. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unable to parse YAML file'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_yaml_error_not_raised. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'replay_dir: /tmp\ncookiecutters_dir: /tmp\n'
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_get_config_file_does_not_exist. Retrieved 1/4 statements.
# Partially parsed test_get_config_valid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 7/12 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 6/13 statements.
# Partially parsed test_get_config_merges_with_defaults. Retrieved 3/6 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_nested_dict_merge. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n'
    var_2 = 'utf-8'

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
    var_0 = 'TEST_REPLAY_DIR'
    var_1 = '/test/replays'
    var_2 = 'TEST_COOKIES_DIR'
    var_3 = '/test/cookies'
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
    var_6 = '/'
    var_7 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /custom/replays\n'
    var_2 = 'utf-8'
    var_3 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  custom_abbr: value\n'
    var_2 = 'utf-8'
    var_3 = 'abbreviations'



