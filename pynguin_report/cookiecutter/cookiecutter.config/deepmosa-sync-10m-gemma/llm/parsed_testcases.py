####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = '/tmp/custom_replay'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = var_3['replay_dir']
    assert var_4 == '/tmp/custom_replay'
    var_5 = var_3['cookiecutters_dir']

def test_case_0():
    pass

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = 'd'
    var_10 = 99
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = {var_1: var_11, var_9: var_12}
    var_14 = {var_3: var_10, var_4: var_6}
    var_15 = {var_0: var_2, var_1: var_14, var_9: var_12}
    var_16 = module_0.merge_configs(var_8, var_13)
    var_17 = bool(var_16 == var_15)
    assert var_17 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_config_success. Retrieved 14/20 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_key'
    var_3 = '/tmp/replay'
    var_4 = '~/cookies'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'config.yaml'
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = 'extra'
    var_10 = '/default/replay'
    var_11 = '/default/cookies'
    var_12 = 'base'
    var_13 = {var_0: var_10, var_1: var_11, var_9: var_12}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'bad.yaml'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_user_config_predicate_false_via_env_var_set. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_success. Retrieved 14/21 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 5/10 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 5/10 statements.
# Partially parsed test_get_config_path_expansion. Retrieved 10/16 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecut_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {}
    var_9 = module_0.dump(var_7, **var_8)
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'cookiecutters_dir'
    var_12 = '/default/path'
    var_13 = '/default/cookies'
    var_14 = {var_1: var_12, var_11: var_13}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'key: : value :'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$TEST_VAR/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dump(var_5, **var_6)
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = ''
    var_10 = {var_1: var_9, var_2: var_9}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 2/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_config_ensures_predicate_is_false. Retrieved 3/10 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.get_config(var_0)
    var_3 = var_2['key']
    assert var_3 == 'value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 2/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.get_user_config(default_config=var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #9
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_config_ensures_predicate_is_false. Retrieved 2/19 statements.


def test_case_0():
    var_0 = '- item1\n- item2'
    var_1 = 'key: value'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_user_config_merges_provided_dict_with_defaults. Retrieved 6/7 statements.
# Partially parsed test_get_user_config_uses_env_var_if_set. Retrieved 3/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'new_key'
    var_2 = '/tmp/test'
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)

def test_case_0():
    pass

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = module_0.get_user_config()
    var_2 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = 'e'
    var_10 = 20
    var_11 = {var_3: var_10}
    var_12 = 5
    var_13 = {var_1: var_11, var_9: var_12}
    var_14 = {var_3: var_10, var_4: var_6}
    var_15 = {var_0: var_2, var_1: var_14, var_9: var_12}
    var_16 = module_0.merge_configs(var_8, var_13)
    var_17 = bool(var_16 == var_15)
    assert var_17 is True



# Parsed testcases at query #12
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {'loaded': True})
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_config_evaluates_predicate_true. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_loads_custom_file_when_path_provided. Retrieved 7/15 statements.
# Partially parsed test_get_user_config_raises_error_on_invalid_env_path. Retrieved 2/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'new_key'
    var_2 = '/tmp/test'
    var_3 = 'new_val'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/tmp/test'
    var_7 = var_5['new_key']
    assert var_7 == 'new_val'
    var_8 = 'cookiecutters_dir'
    var_9 = bool('cookiecutters_dir' in var_5)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = '/custom/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config(var_0)
    var_7 = var_6['replay_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = 'd'
    var_10 = 20
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = {var_1: var_11, var_9: var_12}
    var_14 = module_0.merge_configs(var_8, var_13)
    var_15 = var_14['a']
    assert var_15 == 1
    var_16 = var_14['nested']['b']
    assert var_16 == 20
    var_17 = var_14['nested']['c']
    assert var_17 == 3
    var_18 = var_14['d']
    assert var_18 == 4



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_config_empty_yaml_returns_empty_dict. Retrieved 4/16 statements.


def test_case_0():
    var_0 = False
    var_1 = 'w'
    var_2 = '.yaml'
    var_3 = ''
    var_4 = bool(var_0)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_get_user_config_merges_provided_dict_when_default_config_is_dict. Retrieved 8/9 statements.
# Partially parsed test_get_user_config_loads_custom_config_file. Retrieved 9/23 statements.
# Partially parsed test_get_user_config_uses_env_variable_when_set. Retrieved 7/22 statements.
# Partially parsed test_get_user_config_returns_defaults_when_no_config_found. Retrieved 2/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'new_key'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = 'inner'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.get_user_config(default_config=var_6)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_custom_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/test'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config(var_0)
    var_7 = '/tmp/test'
    var_8 = var_6['replay_dir']
    var_9 = bool(var_6['replay_dir'] == var_3)
    assert var_9 is True
    var_10 = '/tmp/cookies'
    var_11 = var_6['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_env_config.yaml'
    var_1 = 'replay_dir'
    var_2 = '/env/path'
    var_3 = {var_1: var_2}
    var_4 = module_0.get_user_config()
    var_5 = '/env/path'
    var_6 = var_4['replay_dir']
    var_7 = bool(var_4['replay_dir'] == var_2)
    assert var_7 is True
    var_8 = 'COOKIECUTTER_CONFIG'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = module_0.get_user_config()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_config_path_exists. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key: value'
    var_2 = 'utf-8'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_valid_dict_type. Retrieved 6/18 statements.


def test_case_0():
    var_0 = False
    var_1 = 'w'
    var_2 = '.yaml'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(var_0)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_config_evaluates_true_on_empty_file. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = 'w'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_success. Retrieved 22/32 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 17/24 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 17/24 statements.
# Partially parsed test_get_config_empty_file. Retrieved 17/23 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replays'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/path'
    var_10 = '/default/cookies'
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = 'your_module.logger'
    var_13 = 'Logger'
    var_14 = ()
    var_15 = 'debug'
    var_16 = None
    var_17 = lambda self, *args: var_16
    var_18 = {var_15: var_17}
    var_19 = [var_13, var_14, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: : yaml : structure'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 'your_module.logger'
    var_8 = 'Logger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda self, *args: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 'your_module.logger'
    var_8 = 'Logger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda self, *args: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_1.get_config(var_1)

import builtins as module_0

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/def/replays'
    var_6 = '/def/cookies'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'your_module.logger'
    var_9 = 'Logger'
    var_10 = ()
    var_11 = 'debug'
    var_12 = None
    var_13 = lambda self, *args: var_12
    var_14 = {var_11: var_13}
    var_15 = [var_9, var_10, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_config_success. Retrieved 25/31 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 14/20 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 14/20 statements.
# Partially parsed test_get_config_path_expansion. Retrieved 19/26 statements.


import yaml as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutkeys_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {}
    var_9 = module_0.dump(var_7, **var_8)
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'cookiecutters_dir'
    var_12 = 'existing'
    var_13 = '/default/path'
    var_14 = '/default/cookies'
    var_15 = True
    var_16 = {var_1: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'your_module.logger'
    var_18 = 'Mock'
    var_19 = ()
    var_20 = 'debug'
    var_21 = None
    var_22 = lambda *args: var_21
    var_23 = {var_20: var_22}
    var_24 = [var_18, var_19, var_23]
    var_25 = {}
    var_26 = module_1.type(*var_24, **var_25)
    var_27 = var_26()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'key: : value :'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Mock'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Mock'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_0)

import yaml as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'expand.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$TEST_VAR/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dump(var_5, **var_6)
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = ''
    var_10 = {var_1: var_9, var_2: var_9}
    var_11 = 'your_module.logger'
    var_12 = 'Mock'
    var_13 = ()
    var_14 = 'debug'
    var_15 = None
    var_16 = lambda *args: var_15
    var_17 = {var_14: var_16}
    var_18 = [var_12, var_13, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()



# Parsed testcases at query #22
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #23
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'fake_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_success. Retrieved 22/32 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 14/21 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 14/21 statements.
# Partially parsed test_get_config_merges_with_default. Retrieved 20/27 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_setting'
    var_4 = '/tmp/replays'
    var_5 = '~/cookies'
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/replays'
    var_10 = '/default/cookies'
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = 'your_module.logger'
    var_13 = 'Mock'
    var_14 = ()
    var_15 = 'debug'
    var_16 = None
    var_17 = lambda *args: var_16
    var_18 = {var_15: var_17}
    var_19 = [var_13, var_14, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Mock'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Mock'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)

import builtins as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = '/new/path'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutters_dir'
    var_5 = 'extra'
    var_6 = '/old/path'
    var_7 = '/old/cookies'
    var_8 = 1
    var_9 = {var_1: var_6, var_4: var_7, var_5: var_8}
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'your_module.logger'
    var_12 = 'Mock'
    var_13 = ()
    var_14 = 'debug'
    var_15 = None
    var_16 = lambda *args: var_15
    var_17 = {var_14: var_16}
    var_18 = [var_12, var_13, var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_20()



# Parsed testcases at query #25
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_config_success. Retrieved 15/21 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 5/10 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 5/10 statements.
# Partially parsed test_get_config_empty_yaml. Retrieved 8/13 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {}
    var_9 = module_0.dump(var_7, **var_8)
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'base_key'
    var_12 = 'base_val'
    var_13 = '/default/replay'
    var_14 = '/default/cookies'
    var_15 = {var_11: var_12, var_1: var_13, var_2: var_14}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'key: : invalid'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/default/r'
    var_6 = '/default/c'
    var_7 = {var_3: var_5, var_4: var_6}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_config_success. Retrieved 25/31 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 11/13 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 14/20 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 14/20 statements.
# Partially parsed test_get_config_path_expansion. Retrieved 20/29 statements.


import yaml as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutKeys_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {}
    var_9 = module_0.dump(var_7, **var_8)
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'cookiecutters_dir'
    var_12 = 'extra'
    var_13 = '/default/path'
    var_14 = '/default/cookies'
    var_15 = 1
    var_16 = {var_1: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'your_module.logger'
    var_18 = 'Logger'
    var_19 = ()
    var_20 = 'debug'
    var_21 = None
    var_22 = lambda *args: var_21
    var_23 = {var_20: var_22}
    var_24 = [var_18, var_19, var_23]
    var_25 = {}
    var_26 = module_1.type(*var_24, **var_25)
    var_27 = var_26()

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'your_module.logger'
    var_1 = 'Logger'
    var_2 = ()
    var_3 = 'debug'
    var_4 = None
    var_5 = lambda *args: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = var_9()
    var_11 = 'non_existent_path.yaml'
    var_12 = module_1.get_config(var_11)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad_config.yaml'
    var_1 = 'key: : value :'
    var_2 = 'your_module.logger'
    var_3 = 'Logger'
    var_4 = ()
    var_5 = 'debug'
    var_6 = None
    var_7 = lambda *args: var_6
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'your_module.DEFAULT_CONFIG'
    var_14 = {}
    var_15 = module_1.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.logger'
    var_3 = 'Logger'
    var_4 = ()
    var_5 = 'debug'
    var_6 = None
    var_7 = lambda *args: var_6
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = 'your_module.DEFAULT_CONFIG'
    var_14 = {}
    var_15 = module_1.get_config(var_0)

import yaml as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'expand_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$TEST_VAR/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = module_0.dump(var_5, **var_6)
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = ''
    var_10 = {var_1: var_9, var_2: var_9}
    var_11 = 'your_module.logger'
    var_12 = 'Logger'
    var_13 = ()
    var_14 = 'debug'
    var_15 = None
    var_16 = lambda *args: var_15
    var_17 = {var_14: var_16}
    var_18 = [var_12, var_13, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'TEST_VAR'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_get_config_success. Retrieved 13/22 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 8/14 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 10/16 statements.
# Partially parsed test_get_config_expands_env_vars. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/replay'
    var_10 = '/default/cookies'
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'key: : invalid'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = 'item1'
    var_2 = 'item2'
    var_3 = [var_1, var_2]
    var_4 = 'your_module.DEFAULT_CONFIG'
    var_5 = 'replay_dir'
    var_6 = 'cookiecutters_dir'
    var_7 = ''
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'env_test.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$TEST_VAR/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'your_module.DEFAULT_CONFIG'
    var_7 = 'replay_dir'
    var_8 = 'cookiecutters_dir'
    var_9 = ''
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = 'my_folder/replay'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_config_yaml_is_dict. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #30
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'loaded': True})
    assert var_1 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_line_10_evaluates_to_true_with_valid_yaml. Retrieved 6/21 statements.
# Partially parsed test_get_config_line_10_evaluates_to_true_with_empty_file. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'key: value'
    var_1 = 'test_config.yaml'
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '.'

def test_case_0():
    var_0 = 'empty_config.yaml'
    var_1 = ''
    var_2 = 'utf-8'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '.'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_raises_invalid_configuration_on_yaml_error. Retrieved 3/9 statements.


def test_case_0():
    var_0 = ': invalid : yaml'
    var_1 = 'invalid_config.yaml'
    var_2 = 'utf-8'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_get_config_exists_and_opens_file. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_config_file_exists. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_get_config_ensures_yaml_is_dict. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'key: value'
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_get_user_config_merges_provided_dict_with_defaults. Retrieved 4/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'new_value'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)

def test_case_0():
    pass

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/tmp/non_existent_config_file_12345.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(True)
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = 'd'
    var_10 = 20
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = {var_1: var_11, var_9: var_12}
    var_14 = {var_3: var_10, var_4: var_6}
    var_15 = {var_0: var_2, var_1: var_14, var_9: var_12}
    var_16 = module_0.merge_configs(var_8, var_13)
    var_17 = bool(var_16 == var_15)
    assert var_17 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.merge_configs(var_2, var_3)
    var_5 = bool(var_4 == var_2)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_get_config_raises_invalid_configuration_on_yaml_error. Retrieved 7/15 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'invalid: : yaml'
    var_2 = 'safe_load'
    var_3 = 'raise yaml.YAMLError("test error")'
    var_4 = exec(var_3)
    var_5 = lambda x: var_4
    var_6 = module_0.get_config(var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'new_key'
    var_2 = '/tmp/test'
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = var_5['replay_dir']
    assert var_6 == '/tmp/test'
    var_7 = var_5['new_key']
    assert var_7 == 'value'
    var_8 = var_5['existing']
    assert var_8 == 1

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/default'
    var_3 = var_1['existing']
    assert var_3 == 1

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/tmp/custom_config.yaml'
    var_1 = "replay_dir: '/custom/path'\ncookiecutters_dir: '/cookies'"
    var_2 = module_0.get_user_config(var_0)
    var_3 = var_2['replay_dir']
    assert var_3 == '/custom/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/env/config.yaml'
    var_1 = module_0.get_user_config()
    var_2 = var_1['replay_dir']
    assert var_2 == '/env/path'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/default'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/path'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_expand_path_with_env_vars. Retrieved 4/6 statements.
# Partially parsed test_expand_path_with_user_home. Retrieved 4/6 statements.
# Partially parsed test_expand_path_with_both_env_and_home. Retrieved 8/16 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_VAR/file.txt'
    var_1 = module_0._expand_path(var_0)
    var_2 = 'my_folder'
    var_3 = 'file.txt'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '~/documents'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'
    var_3 = 'documents'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '$TEST_DIR/~$USER_HOME/config'
    var_1 = module_0._expand_path(var_0)
    var_2 = '~'
    var_3 = 'data'
    var_4 = 'config'
    var_5 = '~/$SUB'
    var_6 = module_0._expand_path(var_5)
    var_7 = 'sub'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/absolute/path/to/file'
    var_1 = module_0._expand_path(var_0)
    assert var_1 == '/absolute/path/to/file'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._expand_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_config_success. Retrieved 15/28 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 4/12 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 3/12 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 3/12 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new_key'
    var_8 = '$HOME/expanded_replay'
    var_9 = 'new_val'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = 'config.yaml'
    var_12 = var_0 / var_11
    var_13 = str(var_12)
    var_14 = module_0.get_config(var_13)
    var_15 = var_14['replay_dir']
    assert var_15 == '/user/home/expanded_replay'
    var_16 = var_14['cookiecutters_dir']
    assert var_16 == '/tmp/cookies'
    var_17 = var_14['new_key']
    assert var_17 == 'new_val'
    var_18 = var_14['other']
    assert var_18 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'non_existent_file.yaml'
    var_2 = 'non_existent_path_12345.yaml'
    var_3 = module_0.get_config(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = module_0.get_config(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_path_exists. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key: value'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_user_config_loads_custom_file_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_uses_env_variable_if_present. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_falls_back_to_user_config_path_if_no_env_var. Retrieved 4/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 == {'key': 'default_val'})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'new_key'
    var_1 = 'key'
    var_2 = 'new_val'
    var_3 = 'overwritten_val'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = bool(var_5 == {'key': 'overwritten_val', 'other': 'stay', 'new_key': 'new_val'})
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'loaded'
    var_1 = 'from_custom_path'
    var_2 = '/custom/path.yaml'
    var_3 = module_0.get_user_config(var_2)
    var_4 = bool(var_3 == {'loaded': 'from_custom_path'})
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'loaded'
    var_1 = 'from_env'
    var_2 = module_0.get_user_config()
    var_3 = bool(var_2 == {'loaded': 'from_env'})
    assert var_3 is True
    var_4 = '/env/path.yaml'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'loaded'
    var_1 = 'from_user_path'
    var_2 = module_0.get_user_config()
    var_3 = bool(var_2 == {'loaded': 'from_user_path'})
    assert var_3 is True
    var_4 = '/default/user_path.yaml'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'key': 'default'})
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {'loaded': True})
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_get_config_success. Retrieved 13/23 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 5/11 statements.
# Partially parsed test_get_config_not_a_dictionary. Retrieved 5/11 statements.
# Partially parsed test_get_config_empty_file. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_setting'
    var_3 = '/tmp/replays'
    var_4 = '~/templates'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'config.yaml'
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/replays'
    var_10 = '/default/templates'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = '/def'
    var_6 = {var_3: var_5, var_4: var_5}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_get_config_yaml_is_dict. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key: value'
    var_2 = 'utf-8'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_get_config_yaml_is_dict. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '---\nkey: value\n'
    var_2 = 'utf-8'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_get_user_config_keyerror_is_avoided_by_setting_env. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = module_0.get_user_config(var_1, var_0)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 == {'key': 'default_val'})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'new_key'
    var_1 = 'key'
    var_2 = 'new_val'
    var_3 = 'overwritten'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = bool(var_5 == {'key': 'overwritten', 'other': 'stay', 'new_key': 'new_val'})
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/tmp/custom_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cc'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config(var_0)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/env_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cc'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_user_config()
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '/tmp/replay'
    var_3 = '/tmp/cc'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config()
    var_6 = bool(var_5 == var_4)
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config()
    var_4 = bool(var_3 == var_2)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.get_user_config(var_0, var_1)
    var_3 = bool(var_2 == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = bool(var_1 == {'key': 'value', 'replay_dir': '.', 'cookiecutters_dir': '.'})
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_get_config_valid_dict_structure. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_get_config_evaluates_true_on_empty_yaml. Retrieved 1/13 statements.


def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_config_success. Retrieved 22/32 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 14/21 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 14/21 statements.
# Partially parsed test_get_config_empty_file. Retrieved 16/22 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/replay'
    var_10 = '/default/cookies'
    var_11 = {var_1: var_9, var_2: var_10}
    var_12 = 'your_module.logger'
    var_13 = 'Logger'
    var_14 = ()
    var_15 = 'debug'
    var_16 = None
    var_17 = lambda self, *args: var_16
    var_18 = {var_15: var_17}
    var_19 = [var_13, var_14, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path_12345.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Logger'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda self, *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Logger'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda self, *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)

import builtins as module_0

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/def'
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = 'your_module.DEFAULT_CONFIG'
    var_7 = 'your_module.logger'
    var_8 = 'Logger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda self, *args: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_get_config_path_exists. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = './data'
    var_4 = './cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_get_config_success. Retrieved 13/21 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 8/14 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 8/14 statements.
# Partially parsed test_get_config_empty_file. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_key'
    var_3 = '/tmp/replays'
    var_4 = '~/.templates'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'config.yaml'
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = 'base'
    var_10 = 'default'
    var_11 = '/old'
    var_12 = {var_9: var_10, var_0: var_11, var_1: var_11}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'key: : value :'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = 'your_module.DEFAULT_CONFIG'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/default/replays'
    var_5 = '/default/cookies'
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 4/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = False
    var_3 = module_0.get_user_config(var_1, var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_config_success. Retrieved 17/26 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 5/10 statements.
# Partially parsed test_get_config_not_a_dict. Retrieved 5/10 statements.
# Partially parsed test_get_config_empty_file. Retrieved 7/12 statements.


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_setting'
    var_4 = '/tmp/replays'
    var_5 = '~/cookies'
    var_6 = 'value'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {}
    var_9 = module_0.dump(var_7, **var_8)
    var_10 = 'your_module.DEFAULT_CONFIG'
    var_11 = 'extra'
    var_12 = '/default/replays'
    var_13 = '/default/cookies'
    var_14 = 'orig'
    var_15 = {var_1: var_12, var_2: var_13, var_11: var_14}
    var_16 = '~'
    var_17 = '/tmp/cookies'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'key: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = ''
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/default'
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = 'your_module.DEFAULT_CONFIG'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_get_config_raises_invalid_configuration_on_yaml_error. Retrieved 7/14 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: : yaml'
    var_2 = 'safe_load'
    var_3 = 'raise yaml.YAMLError("Parse Error")'
    var_4 = exec(var_3)
    var_5 = lambda x: var_4
    var_6 = module_0.get_config(var_0)



# Parsed testcases at query #25
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'fake_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_loads_custom_file_when_path_provided. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_uses_env_var_when_set. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_falls_back_to_default_path_when_no_env_var. Retrieved 4/10 statements.
# Partially parsed test_get_user_config_returns_defaults_when_no_env_and_no_user_file. Retrieved 1/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 == {'key': 'value'})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = 'c'
    var_2 = 3
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = bool(var_5 == {'a': 1, 'b': 3, 'c': 4})
    assert var_6 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'loaded'
    var_1 = True
    var_2 = 'custom_path.yaml'
    var_3 = module_0.get_user_config(var_2)
    var_4 = bool(var_3 == {'loaded': True})
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'from_env'
    var_1 = True
    var_2 = module_0.get_user_config()
    var_3 = bool(var_2 == {'from_env': True})
    assert var_3 is True
    var_4 = '/env/path.yaml'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'your_module.USER_CONFIG_PATH'
    var_1 = 'from_user_path'
    var_2 = True
    var_3 = module_0.get_user_config()
    var_4 = bool(var_3 == {'from_user_path': True})
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'default': 'val'})
    assert var_1 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_get_user_config_merges_provided_dict_when_default_config_is_dict. Retrieved 12/15 statements.
# Partially parsed test_get_user_config_loads_custom_config_file. Retrieved 4/9 statements.
# Partially parsed test_get_user_config_uses_env_var_when_present. Retrieved 5/11 statements.
# Partially parsed test_get_user_config_loads_user_config_path_if_exists. Retrieved 4/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 == {'key': 'value'})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = 'c'
    var_2 = 3
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 1
    var_7 = 2
    var_8 = {var_5: var_6, var_0: var_7}
    var_9 = {var_0: var_2, var_1: var_3}
    var_10 = {var_0: var_2, var_1: var_3}
    var_11 = module_0.get_user_config(default_config=var_10)
    var_12 = bool(var_11 == {'a': 1, 'b': 3, 'c': 4})
    assert var_12 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/fake/path/config.yaml'
    var_1 = 'loaded'
    var_2 = 'true'
    var_3 = module_0.get_user_config(var_0)
    var_4 = bool(var_3 == {'loaded': 'true'})
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'from'
    var_1 = 'env'
    var_2 = None
    var_3 = module_0.get_user_config(var_2)
    var_4 = bool(var_3 == {'from': 'env'})
    assert var_4 is True
    var_5 = '/env/path/config.yaml'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(var_1 == {'default': 'val'})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'status'
    var_1 = 'found'
    var_2 = None
    var_3 = module_0.get_user_config(var_2)
    var_4 = bool(var_3 == {'status': 'found'})
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_get_config_success. Retrieved 21/28 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 12/18 statements.
# Partially parsed test_get_config_not_a_dictionary. Retrieved 12/18 statements.
# Partially parsed test_get_config_empty_file. Retrieved 16/22 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_key'
    var_3 = '/tmp/replays'
    var_4 = '~/.cookiecutters'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'config.yaml'
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = '/default/replays'
    var_10 = '/default/cookies'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = 'your_module.logger'
    var_13 = 'MockLogger'
    var_14 = ()
    var_15 = 'debug'
    var_16 = None
    var_17 = lambda *a, **k: var_16
    var_18 = {var_15: var_17}
    var_19 = [var_13, var_14, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = var_21()
    var_23 = 'cookiecutters_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.logger'
    var_3 = 'MockLogger'
    var_4 = ()
    var_5 = 'debug'
    var_6 = None
    var_7 = lambda *a, **k: var_6
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.logger'
    var_3 = 'MockLogger'
    var_4 = ()
    var_5 = 'debug'
    var_6 = None
    var_7 = lambda *a, **k: var_6
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = module_1.get_config(var_1)

import builtins as module_0

def test_case_0():
    var_0 = 'empty.yaml'
    var_1 = 'your_module.DEFAULT_CONFIG'
    var_2 = 'replay_dir'
    var_3 = 'cookiecutters_dir'
    var_4 = '/default/replays'
    var_5 = '/default/cookies'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'your_module.logger'
    var_8 = 'MockLogger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda *a: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_get_config_file_exists_and_is_openable. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '/tmp/replay'
    var_4 = '/tmp/cookies'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_get_config_success. Retrieved 23/30 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 14/21 statements.
# Partially parsed test_get_config_non_dict_yaml. Retrieved 14/21 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_setting'
    var_4 = '/tmp/replay'
    var_5 = '~/cookies'
    var_6 = 123
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = 'base'
    var_10 = '/default/replay'
    var_11 = '/default/cookies'
    var_12 = True
    var_13 = {var_1: var_10, var_2: var_11, var_9: var_12}
    var_14 = 'your_module.logger'
    var_15 = 'Logger'
    var_16 = ()
    var_17 = 'debug'
    var_18 = None
    var_19 = lambda *args: var_18
    var_20 = {var_17: var_19}
    var_21 = [var_15, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Logger'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = {}
    var_4 = 'your_module.logger'
    var_5 = 'Logger'
    var_6 = ()
    var_7 = 'debug'
    var_8 = None
    var_9 = lambda *args: var_8
    var_10 = {var_7: var_9}
    var_11 = [var_5, var_6, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = module_1.get_config(var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_get_config_success. Retrieved 23/31 statements.
# Partially parsed test_get_config_file_not_found. Retrieved 2/5 statements.
# Partially parsed test_get_config_invalid_yaml. Retrieved 17/24 statements.
# Partially parsed test_get_config_top_level_not_dict. Retrieved 17/24 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_setting'
    var_3 = '~/replays'
    var_4 = '/tmp/cookies'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'config.yaml'
    var_8 = 'your_module.DEFAULT_CONFIG'
    var_9 = 'extra'
    var_10 = '/default/path'
    var_11 = '/default/cookies'
    var_12 = 1
    var_13 = {var_0: var_10, var_1: var_11, var_9: var_12}
    var_14 = 'your_module.logger'
    var_15 = 'Logger'
    var_16 = ()
    var_17 = 'debug'
    var_18 = None
    var_19 = lambda self, *args: var_18
    var_20 = {var_17: var_19}
    var_21 = [var_15, var_16, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'missing.yaml'
    var_1 = module_0.get_config(var_0)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'bad.yaml'
    var_1 = 'invalid: [unclosed bracket'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 'your_module.logger'
    var_8 = 'Logger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda self, *args: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_1.get_config(var_1)

import builtins as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'list.yaml'
    var_1 = '- item1\n- item2'
    var_2 = 'your_module.DEFAULT_CONFIG'
    var_3 = 'replay_dir'
    var_4 = 'cookiecutters_dir'
    var_5 = ''
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = 'your_module.logger'
    var_8 = 'Logger'
    var_9 = ()
    var_10 = 'debug'
    var_11 = None
    var_12 = lambda self, *args: var_11
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]
    var_15 = {}
    var_16 = module_0.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_1.get_config(var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_get_config_evaluates_predicate_to_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'Ensures that the predicate at line 14 (not isinstance(yaml_dict, dict)) evaluates to False.'
    var_1 = 'test_config.yaml'
    var_2 = 'key: value'
    var_3 = 'utf-8'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_evaluates_true_on_empty_yaml. Retrieved 1/13 statements.


def test_case_0():
    var_0 = ''
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_user_config_predicate_false. Retrieved 1/3 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #35
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'b'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = module_0.get_user_config(default_config=var_2)
    var_4 = bool(var_3 == {'a': 1, 'b': 3})
    assert var_4 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = bool(var_1 == {'a': 1})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/tmp/custom.yaml'
    var_1 = module_0.get_user_config(var_0)
    var_2 = bool(var_1 == {'loaded': True})
    assert var_2 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'env': True})
    assert var_1 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'user': True})
    assert var_1 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = bool(var_0 == {'default': 'val'})
    assert var_1 is True



