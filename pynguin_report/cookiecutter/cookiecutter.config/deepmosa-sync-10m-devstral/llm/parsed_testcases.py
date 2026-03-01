####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/path'

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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/invalid/config.yaml'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/nonexistent/config.yaml'
    var_1 = module_0.get_user_config(var_0)



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_default_path. Retrieved 5/9 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/8 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/8 statements.
# Partially parsed test_get_config_merges_with_default_config. Retrieved 10/14 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 8/14 statements.


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
    var_1 = '- list item'
    var_2 = 'utf-8'

import yaml as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'nested_key'
    var_4 = 'nested_value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'test_config.yaml'
    var_8 = {}
    var_9 = module_0.dump(var_6, **var_8)
    var_10 = 'utf-8'

import yaml as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = '~/test_dir'
    var_3 = '$HOME/test_dir'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_config.yaml'
    var_6 = {}
    var_7 = module_0.dump(var_4, **var_6)
    var_8 = 'utf-8'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 3/5 statements.


import codecs as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'utf-8'
    var_2 = module_0.open(var_0, encoding=var_1)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_43. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_yaml_safe_load_returns_none.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_yaml_error_handling. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)
    var_3 = str(var_2)
    var_4 = bool(var_3 == f'Unable to parse YAML file {var_0}.')
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_config_raises_invalid_configuration_when_yaml_dict_is_not_dict. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/valid/file.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = 'Top-level element of YAML file path/to/valid/file.yaml should be an object.'
    var_4 = bool('Top-level element of YAML file path/to/valid/file.yaml should be an object.' in var_2)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_file.yaml'



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

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'path/to/invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file path/to/invalid.yaml.'



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_config_path_does_not_exist. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_config_path'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


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
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_config_path_exists_and_is_openable. Retrieved 3/5 statements.


import codecs as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'utf-8'
    var_2 = module_0.open(var_0, encoding=var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/11 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 4/6 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 4/6 statements.


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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_env_vars.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/.replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/.cookiecutters'
    var_5 = var_1['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_user_home.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '~/.replay'
    var_3 = var_1['replay_dir']
    var_4 = '~/.cookiecutters'
    var_5 = var_1['cookiecutters_dir']



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var_config. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 2/5 statements.


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
    var_3 = var_1['key']
    assert var_3 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = var_0['key']
    assert var_3 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = var_0['key']
    assert var_2 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_config_path_does_not_exist. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/non/existent/path'



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_yaml_safe_load_returns_none.




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

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'key: value'
    var_1 = ''



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/6 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_default_and_yaml_configs. Retrieved 7/11 statements.
# Partially parsed test_get_config_expands_environment_variables_and_user_home. Retrieved 9/15 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: [unclosed'
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
    var_2 = 'cookiecutters_dir'
    var_3 = '$HOME/test'
    var_4 = '$USER/test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)
    var_7 = var_6['replay_dir']
    var_8 = var_6['cookiecutters_dir']
    var_9 = var_6['other_default_key']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_expand.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$HOME/test'
    var_4 = '~/test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.get_config(var_0)
    var_7 = '~'
    var_8 = '/test'
    var_9 = var_6['replay_dir']
    var_10 = var_6['cookiecutters_dir']



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_get_config_with_valid_yaml. Retrieved 5/10 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/7 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/7 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n    replay_dir: ~/test_replay\n    cookiecutters_dir: ~/test_cookies\n    '
    var_1 = 'test_config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = '~/test_replay'
    var_4 = var_2['replay_dir']
    var_5 = '~/test_cookies'
    var_6 = var_2['cookiecutters_dir']

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid yaml content'
    var_1 = 'invalid_config.yaml'
    var_2 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '- not a dict'
    var_1 = 'non_dict_config.yaml'
    var_2 = module_0.get_config(var_1)



# Parsed testcases at query #35
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookies'
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_config_expands_environment_variables. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/fixtures/invalid.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/fixtures/not_dict.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/fixtures/valid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = var_1['abbreviations']
    var_3 = bool(var_1['abbreviations'] == {'nf': 'notebooks/'})
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/fixtures/with_env_var.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/test/dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/fixtures/with_home.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '~/.cookiecutters'
    var_3 = var_1['cookiecutters_dir']



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #3
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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_path'
    var_1 = module_0.get_user_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = module_0.get_user_config(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 12/17 statements.


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
    var_9 = '$HOME/replay'
    var_10 = '$HOME/cookiecutters'
    var_11 = module_0.get_config(var_0)

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



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_keyerror_exception_raised. Retrieved 3/4 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #22
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_config_raises_exception_when_yaml_is_invalid. Retrieved 3/6 statements.
# Partially parsed test_get_config_raises_exception_when_yaml_top_level_is_not_dict. Retrieved 3/6 statements.
# Partially parsed test_get_config_merges_with_default_and_expands_paths. Retrieved 9/13 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid yaml content'
    var_1 = 'invalid.yaml'
    var_2 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = '- list item'
    var_1 = 'not_dict.yaml'
    var_2 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'cookiecutters_dir'
    var_2 = 'other_setting'
    var_3 = '~/test_replay'
    var_4 = '$HOME/test_cookies'
    var_5 = 'value'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'test_config.yaml'
    var_8 = module_0.get_config(var_7)
    var_9 = var_8['replay_dir']
    var_10 = var_8['cookiecutters_dir']
    var_11 = var_8['other_setting']
    assert var_11 == 'value'
    var_12 = var_8['default_setting']



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'existing_config_path.yaml'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_get_config_with_valid_yaml_file. Retrieved 10/12 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/6 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/6 statements.


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
    var_8 = 'replay_dir: $HOME/replay_dir\ncookiecutters_dir: $HOME/cookiecutters_dir\nother_key: other_value'
    var_9 = module_0.get_config(var_0)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- not a dict'
    var_2 = module_0.get_config(var_0)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_keyerror_predicate.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/7 statements.


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

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #16
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
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/7 statements.


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



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #21
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
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 3/6 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 2/6 statements.


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
    var_3 = var_1['key']
    assert var_3 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'env_path'
    var_2 = 'utf-8'
    var_3 = var_0['key']
    assert var_3 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = 'utf-8'
    var_2 = var_0['key']
    assert var_2 == 'value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 6/10 statements.
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



# Parsed testcases at query #4
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

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #7
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #10
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
    var_9 = '/expanded/path'
    var_10 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_9}
    var_11 = module_0.get_config(var_0)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 3/6 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_2 == {'key': 'value'})
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------




import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = module_0.open(var_0)
    var_2 = module_1.safe_load(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_file.yaml'



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_14_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_43. Retrieved 2/5 statements.


def test_case_0():
    var_0 = True
    var_1 = '/some/path'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_yaml_dict_is_not_a_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_config_path_exists_and_is_file. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file tests/invalid_yaml.yaml.'



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_get_config_raises_ConfigDoesNotExistException. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_config.yaml'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_yaml_error_raised_when_invalid_yaml. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = 'Unable to parse YAML file'
    var_4 = bool('Unable to parse YAML file' in var_2)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_config_path_is_not_a_file. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_config_path'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'valid_config.yaml'



# Parsed testcases at query #32
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)



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



# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = None
    var_1 = bool(not (var_0 or {}))
    assert var_1 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_config_file_opens_successfully. Retrieved 3/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'key: value'
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_2 == {'key': 'value', 'replay_dir': '/path', 'cookiecutters_dir': '/path'})
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


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
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/configs/invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file tests/configs/invalid_yaml.yaml.'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #40
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



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_yaml_safe_load_returns_none.




# Parsed testcases at query #43
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml_file.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none_or_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #46
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

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_config_path_does_not_exist. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_config_path.yaml'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_at_line_43. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_config_file_not_found_in_environment. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #51
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
    var_0 = '/custom/config'
    var_1 = module_0.get_user_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_yaml_dict_is_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #53
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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_paths. Retrieved 4/8 statements.


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
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_paths.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 8/11 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '~/.local/share/replay'
    var_5 = '~/.local/share/cookiecutters'
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



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_yaml_dict_is_not_a_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_yaml_dict_is_not_a_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #60
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
    var_0 = 'invalid_config.yaml'
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



# Parsed testcases at query #61
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



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_at_line_43. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 4/6 statements.


import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'empty.yaml'
    var_2 = module_0.open(var_1)
    var_3 = module_1.safe_load(var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_env_config_file. Retrieved 3/4 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 3/5 statements.
# Partially parsed test_get_user_config_with_no_config_file. Retrieved 4/6 statements.


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
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'env_config.yaml'
    var_1 = module_0.get_user_config()
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
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
    var_2 = False
    var_3 = module_0.get_user_config()



# Parsed testcases at query #65
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
    var_2 = var_1['replay_dir']
    assert var_2 == '/custom/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/env/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()
    var_1 = var_0['replay_dir']
    assert var_1 == '/user/replay'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = module_0.get_user_config()



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
    var_0 = 'valid_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid: yaml: content: ['
    var_1 = 'invalid.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = str(var_1)
    assert var_3 == 'Unable to parse YAML file invalid.yaml.'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_path_does_not_exist. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_path.yaml'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_get_config_returns_merged_dict_with_expanded_paths. Retrieved 4/9 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yml'
    var_1 = module_0.get_config(var_0)
    var_2 = '$HOME/replay'
    var_3 = var_1['replay_dir']
    var_4 = '$HOME/cookiecutters'
    var_5 = var_1['cookiecutters_dir']



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #72
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
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_yaml.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/invalid_yaml.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    assert var_2 == 'Unable to parse YAML file tests/data/invalid_yaml.yaml.'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_config_path_exists_and_is_readable. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'
    var_1 = 'key: value'
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_yaml_safe_load_returns_none. Retrieved 2/5 statements.


def test_case_0():
    var_0 = ''
    var_1 = {}



# Parsed testcases at query #77
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_isinstance_check_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_get_config_with_valid_path. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_environment_variables. Retrieved 2/3 statements.
# Partially parsed test_get_config_expands_user_home. Retrieved 3/4 statements.


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
    var_0 = 'non_dict_yaml_config.yaml'
    var_1 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_env_var.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = var_1['replay_dir']
    assert var_2 == 'test_value'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config_with_home.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '~/.cookiecutters'
    var_3 = var_1['cookiecutters_dir']



# Parsed testcases at query #80
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml content'
    var_1 = module_0.safe_load(var_0)



# Parsed testcases at query #81
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_existent_path.yaml'
    var_1 = module_0.get_config(var_0)



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 3/5 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = module_0.get_user_config()



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_yaml_dict_is_not_dict. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #84
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



# Parsed testcases at query #85
#--------------------------




import codecs as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = module_0.open(var_0)
    var_2 = module_1.safe_load(var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'



# Parsed testcases at query #87
#--------------------------




import yaml as module_0

def test_case_0():
    var_0 = 'invalid yaml content'
    var_1 = module_0.safe_load(var_0)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_yaml_safe_load_returns_dict_or_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var_set. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_exists. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_env_var_not_set_and_user_config_not_exists. Retrieved 1/4 statements.


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



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_yaml_error_raises_invalid_configuration. Retrieved 4/8 statements.


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid.yaml'
    var_1 = 'invalid: yaml: content: ['
    var_2 = module_0.get_config(var_0)
    var_3 = str(var_2)
    var_4 = 'Unable to parse YAML file'
    var_5 = bool('Unable to parse YAML file' in var_3)
    assert var_5 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_get_config_with_nonexistent_path. Retrieved 3/6 statements.
# Partially parsed test_get_config_with_invalid_yaml. Retrieved 3/8 statements.
# Partially parsed test_get_config_with_non_dict_yaml. Retrieved 3/8 statements.


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
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_1)
    var_3 = bool(var_2 == f'Config file {var_0} does not exist.')
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_1)
    var_3 = bool(var_2 == f'Unable to parse YAML file {var_0}.')
    assert var_3 is True

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_1)
    var_3 = bool(var_2 == f'Top-level element of YAML file {var_0} should be an object.')
    assert var_3 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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

# Partially parsed test_keyerror_raised_when_env_var_not_set. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_keyerror_predicate.




# Parsed testcases at query #5
#--------------------------

# Failed to parse test_keyerror_predicate.




# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = 'other_key'
    var_4 = '/expanded/replay'
    var_5 = '/expanded/cookies'
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_get_user_config_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_custom_config_file. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_env_var_set. Retrieved 6/10 statements.
# Partially parsed test_get_user_config_env_var_not_set_user_config_exists. Retrieved 5/9 statements.


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



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_predicate_at_line_43.




# Parsed testcases at query #11
#--------------------------

# Failed to parse test_config_file_exists_returns_true.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'path/to/existing/config.yaml'



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_at_line_43_evaluates_to_true.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/8 statements.
# Partially parsed test_get_user_config_with_default_path. Retrieved 4/7 statements.
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



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #21
#--------------------------




import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'custom_path'
    var_1 = module_0.get_user_config(var_0)
    var_2 = module_0.get_config(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_config_path_exists. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'existing_config.yaml'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_43_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = True
    var_3 = '{}'



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_40_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'COOKIECUTTER_CONFIG'
    var_1 = None
    var_2 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_config_path_exists.




# Parsed testcases at query #31
#--------------------------

# Failed to parse test_predicate_at_line_40_evaluates_to_false.




# Parsed testcases at query #32
#--------------------------

# Failed to parse test_user_config_path_exists.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_get_user_config_with_default_config_true. Retrieved 2/3 statements.
# Partially parsed test_get_user_config_with_default_config_dict. Retrieved 4/5 statements.
# Partially parsed test_get_user_config_with_custom_config_file. Retrieved 5/8 statements.
# Partially parsed test_get_user_config_with_env_var. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_user_config_path. Retrieved 4/7 statements.
# Partially parsed test_get_user_config_with_no_config_found. Retrieved 1/4 statements.


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



