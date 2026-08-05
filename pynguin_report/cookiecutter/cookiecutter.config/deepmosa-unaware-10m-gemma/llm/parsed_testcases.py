####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = 'new_key'
    var_4 = '/tmp/custom_replay/'
    var_5 = 'new_val'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = "cookiecutters_dir: '/tmp/custom_cc/'\nabbreviations:\n  new: 'http://new.com/{0}'"
    var_8 = 'config.yaml'
    var_9 = '/tmp/custom_cc/'
    var_10 = 'non_existent_path.yaml'
    var_11 = module_0.get_user_config(var_10)
    var_12 = 'env_config.yaml'
    var_13 = "replay_dir: '/tmp/env_replay/'"
    var_14 = module_0.get_user_config()
    var_15 = '/tmp/env_replay/'
    var_16 = module_0.get_user_config()
    var_17 = "cookiecutters_dir: '/tmp/user_cc/'"
    var_18 = module_0.get_user_config()
    var_19 = '/tmp/user_cc/'
    var_20 = 'key: : value'
    var_21 = 'bad_config.yaml'
    var_22 = module_0.get_user_config(var_19)



# Parsed testcases at query #2
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'cookiecutments_dir'
    var_2 = 'abbreviations'
    var_3 = '/tmp/custom_dir'
    var_4 = 'gh'
    var_5 = 'https://github.com/custom'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'config.yaml'
    var_9 = module_0.dump(var_7)
    var_10 = 'cookiecutters_dir'
    var_11 = 'custom_dir'
    var_12 = 'invalid.yaml'
    var_13 = 'key: : value'
    var_14 = 'list.yaml'
    var_15 = '- item1\n- item2'
    var_16 = 'empty.yaml'
    var_17 = ''
    var_18 = 'replay_dir'

import yaml as module_0

def test_case_0():
    var_0 = 'env_test.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = '$HOME/my_cookies'
    var_4 = '~/my_replay'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dump(var_5)



# Parsed testcases at query #3
#--------------------------


import yaml as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'invalid.yaml'
    var_2 = 'key: : value'
    var_3 = 'utf-8'
    var_4 = 'list.yaml'
    var_5 = '- item1\n- item2'
    var_6 = 'valid.yaml'
    var_7 = 'cookiecutters_dir'
    var_8 = 'abbreviations'
    var_9 = '/tmp/custom_cookies/'
    var_10 = 'new_abbr'
    var_11 = 'https://example.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = module_0.dump(var_13)
    var_15 = 'default_context'
    var_16 = 'env_var.yaml'
    var_17 = 'replay_dir'
    var_18 = '$HOME/replay_test/'
    var_19 = {var_17: var_18}
    var_20 = module_0.dump(var_19)
    var_21 = 'key: value'
    var_22 = 'dummy_path.yaml'
    var_23 = module_1.get_config(var_22)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'abbreviations'
    var_3 = '/tmp/custom_cookies/'
    var_4 = 'new_abbr'
    var_5 = 'https://example.com/{0}'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'config.yaml'
    var_9 = 'invalid.yaml'
    var_10 = 'key: : invalid_syntax'
    var_11 = 'list.yaml'
    var_12 = '- item1\n- item2'
    var_13 = 'empty.yaml'
    var_14 = ''
    var_15 = 'env_var.yaml'
    var_16 = "cookiecutters_dir: '$TEST_DIR/cookies/'"
    var_17 = '/tmp/test_expansion/cookies/'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import yaml as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'invalid.yaml'
    var_2 = 'key: : value'
    var_3 = 'utf-8'
    var_4 = 'list.yaml'
    var_5 = '- item1\n- item2'
    var_6 = 'valid.yaml'
    var_7 = 'replay_dir'
    var_8 = 'abbreviations'
    var_9 = 'some_new_key'
    var_10 = '/tmp/custom_replay'
    var_11 = 'new_abbr'
    var_12 = 'https://example.com/{0}'
    var_13 = {var_11: var_12}
    var_14 = 'some_value'
    var_15 = {var_7: var_10, var_8: var_13, var_9: var_14}
    var_16 = module_0.dump(var_15)
    var_17 = 'cookiecutters_dir'
    var_18 = 'key: value'
    var_19 = 'fake_path.yaml'
    var_20 = module_1.get_config(var_19)



# Parsed testcases at query #2
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'abbreviations'
    var_3 = '/tmp/custom_cookies'
    var_4 = 'gh'
    var_5 = 'https://new-github.com/{0}.git'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'valid_config.yaml'
    var_9 = module_0.dump(var_7)
    var_10 = 'invalid_syntax.yaml'
    var_11 = 'key: [unclosed bracket'
    var_12 = 'list_config.yaml'
    var_13 = '- item1\n- item2'
    var_14 = 'empty.yaml'
    var_15 = ''
    var_16 = 'replay_dir'
    var_17 = '$HOME/custom_replay'
    var_18 = {var_16: var_17}
    var_19 = 'env_expand.yaml'
    var_20 = module_0.dump(var_18)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config with various valid and invalid YAML contents.'
    var_1 = '/fake/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises error when file does not exist.'
    var_1 = '/non/existent/path.yaml'
    var_2 = module_0.get_config(var_1)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises error on invalid YAML syntax.'
    var_1 = 'key: : value'
    var_2 = '/fake/config.yaml'
    var_3 = module_0.get_config(var_2)

import yaml as module_0
import cookiecutter.config as module_1

def test_case_0():
    var_0 = 'Test that get_config expands environment variables and user home.'
    var_1 = 'replay_dir'
    var_2 = 'cookiecutters_dir'
    var_3 = '$HOME/custom_replay'
    var_4 = '~/custom_cookies'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.dump(var_5)
    var_7 = '$HOME'
    var_8 = '/home/user'
    var_9 = '~'
    var_10 = '/fake/config.yaml'
    var_11 = module_1.get_config(var_10)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'non_existent.yaml'
    var_1 = 'invalid.yaml'
    var_2 = 'key: : value'
    var_3 = 'utf-8'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = 'new_key'
    var_4 = '/tmp/replay/'
    var_5 = 'value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.get_user_config(default_config=var_6)
    var_8 = "cookiecutters_dir: /custom/path/\nabbreviations:\n  new: 'https://new.com/{0}'"
    var_9 = '/custom/config.yaml'
    var_10 = module_0.get_user_config(var_9)
    var_11 = '/custom/path/'
    var_12 = module_0.get_user_config()
    var_13 = '/env/replay/'
    var_14 = module_0.get_user_config()
    var_15 = module_0.get_user_config()
    var_16 = '/user/config/'
    var_17 = module_0.get_user_config()
    var_18 = module_0.get_user_config()
    var_19 = module_0.get_user_config()



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = 'new_key'
    var_4 = '/tmp/replay'
    var_5 = 'value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = "abbreviations:\n  custom: 'https://custom.com/{0}'\nreplay_dir: '$HOME/test_replay'"
    var_8 = '/fake/path/config.yaml'
    var_9 = module_0.get_user_config(var_8)
    var_10 = '$HOME/test_replay'
    var_11 = "cookiecutters_dir: '/custom/dir'"
    var_12 = module_0.get_user_config()
    var_13 = '/custom/dir'
    var_14 = module_0.get_user_config()
    var_15 = "replay_dir: '/user/replay'"
    var_16 = module_0.get_user_config()
    var_17 = '/user/replay'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'replay_dir'
    var_3 = 'new_key'
    var_4 = '/tmp/replay'
    var_5 = 'value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.get_user_config(default_config=var_6)
    var_8 = 'custom_config.yaml'
    var_9 = "cookiecutters_dir: '/tmp/custom_dir'\nabbreviations:\n  new: 'https://new.com/{0}'"
    var_10 = '/tmp/custom_dir'
    var_11 = 'non_existent_path.yaml'
    var_12 = module_0.get_user_config(var_11)
    var_13 = 'env_config.yaml'
    var_14 = "replay_dir: '/tmp/env_replay'"
    var_15 = module_0.get_user_config()
    var_16 = '/tmp/env_replay'
    var_17 = module_0.get_user_config()
    var_18 = '/tmp/user_dir'
    var_19 = module_0.get_user_config()
    var_20 = 'invalid.yaml'
    var_21 = 'key: : value'
    var_22 = module_0.get_user_config(var_18)
    var_23 = 'not_a_dict.yaml'
    var_24 = '- item1\n- item2'
    var_25 = module_0.get_user_config(var_18)



