####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = '/nonexistent/path'
    var_3 = module_0.get_config(var_2)
    var_4 = 'invalid: yaml: file\n'
    var_5 = module_0.get_config(var_4)
    var_6 = '- item1\n- item2\n'
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\n'
    var_1 = 'test_config.yml'
    var_2 = module_0.get_config(var_1)
    var_3 = '~/.cookiecutters/'
    var_4 = '~/.cookiecutter_replay/'
    var_5 = 'non_existent_config.yml'
    var_6 = module_0.get_config(var_5)
    var_7 = 'invalid yaml'
    var_8 = 'invalid_config.yml'
    var_9 = module_0.get_config(var_8)
    var_10 = 'invalid_config.yml'
    var_11 = '- item1\n- item2'
    var_12 = 'not_dict_config.yml'
    var_13 = module_0.get_config(var_12)
    var_14 = 'not_dict_config.yml'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /custom/path\n'
    var_1 = 'replay_dir: /another/path\n'
    var_2 = 'invalid yaml content'
    var_3 = '/non/existent/path'
    var_4 = module_0.get_config(var_3)
    var_5 = 'just a string'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '/path/to/invalid/config.yaml'
    var_3 = 'invalid yaml'
    var_4 = module_0.get_config(var_2)
    var_5 = '/path/to/valid/config.yaml'
    var_6 = 'cookiecutters_dir: /path/to/cookiecutters'
    var_7 = module_0.get_config(var_5)
    var_8 = '/path/to/cookiecutters'
    var_9 = '/path/to/config_with_env.yaml'
    var_10 = 'cookiecutters_dir: $TEST_ENV/cookiecutters'
    var_11 = module_0.get_config(var_9)
    var_12 = '/path/to/test/cookiecutters'
    var_13 = '/path/to/config_with_home.yaml'
    var_14 = 'cookiecutters_dir: ~/cookiecutters'
    var_15 = module_0.get_config(var_13)
    var_16 = '~/cookiecutters'
    var_17 = '/path/to/config_with_nested.yaml'
    var_18 = 'abbreviations:\n  gh: https://github.com/{0}.git'
    var_19 = module_0.get_config(var_17)
    var_20 = '/path/to/config_empty.yaml'
    var_21 = ''
    var_22 = module_0.get_config(var_20)
    var_23 = '/path/to/config_non_dict.yaml'
    var_24 = '- item1\n- item2'
    var_25 = module_0.get_config(var_23)
    var_26 = '/path/to/invalid/config.yaml'
    var_27 = '/path/to/valid/config.yaml'
    var_28 = '/path/to/config_with_env.yaml'
    var_29 = '/path/to/config_with_home.yaml'
    var_30 = '/path/to/config_with_nested.yaml'
    var_31 = '/path/to/config_empty.yaml'
    var_32 = '/path/to/config_non_dict.yaml'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '/custom/cookiecutters'
    var_5 = '/custom/replay'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}.git'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'cookiecutters_dir'
    var_14 = 'replay_dir'
    var_15 = 'default_context'
    var_16 = 'abbreviations'
    var_17 = '/custom/cookiecutters'
    var_18 = '/custom/replay'
    var_19 = 'key'
    var_20 = 'value'
    var_21 = (var_19, var_20)
    var_22 = [var_21]
    var_23 = 'gh'
    var_24 = 'gl'
    var_25 = 'bb'
    var_26 = 'custom'
    var_27 = 'https://github.com/{0}.git'
    var_28 = 'https://gitlab.com/{0}.git'
    var_29 = 'https://bitbucket.org/{0}'
    var_30 = 'https://custom.com/{0}.git'
    var_31 = {var_23: var_27, var_24: var_28, var_25: var_29, var_26: var_30}
    var_32 = {var_13: var_17, var_14: var_18, var_15: var_10, var_16: var_31}
    var_33 = 'invalid: yaml: here'
    var_34 = '/non/existent/path'
    var_35 = module_0.get_config(var_34)
    var_36 = 'not a dict'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_3 = 'replay_dir: /tmp/replay\n'
    var_4 = 'default_context:\n'
    var_5 = '  key1: value1\n'
    var_6 = 'abbreviations:\n'
    var_7 = '  gh: https://github.com/{0}.git\n'
    var_8 = '/nonexistent/path'
    var_9 = module_0.get_config(var_8)
    var_10 = 'invalid.yml'
    var_11 = 'invalid: yaml: file'
    var_12 = 'non_dict.yml'
    var_13 = '- item1\n- item2'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '/custom/cookiecutters'
    var_6 = '/custom/replay'
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'valid_config.yml'
    var_2 = 'cookiecutters_dir: /tmp/custom_cookiecutters\n'
    var_3 = 'replay_dir: /tmp/custom_replay\n'
    var_4 = 'default_context:\n'
    var_5 = '  key1: value1\n'
    var_6 = '  key2: value2\n'
    var_7 = 'abbreviations:\n'
    var_8 = '  gh: "https://github.com/{0}.git"\n'
    var_9 = 'nonexistent.yml'
    var_10 = 'invalid.yml'
    var_11 = 'invalid: yaml: file'
    var_12 = 'non_dict.yml'
    var_13 = '- item1\n- item2\n'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'config.yml'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_2 = 'replay_dir: /tmp/replay\n'
    var_3 = 'default_context:\n'
    var_4 = '  key1: value1\n'
    var_5 = '  key2: value2\n'
    var_6 = 'abbreviations:\n'
    var_7 = '  gh: https://github.com/{0}.git\n'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = 'abbreviations'
    var_6 = '/custom/cookiecutters'
    var_7 = '/custom/replay'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'custom'
    var_12 = 'https://custom.com/{0}.git'
    var_13 = {var_11: var_12}
    var_14 = {var_2: var_6, var_3: var_7, var_4: var_10, var_5: var_13}
    var_15 = 'invalid: yaml: here'
    var_16 = '- item1\n- item2'
    var_17 = 'nonexistent.yml'
    var_18 = module_0.get_config(var_1)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = '/custom/cookiecutters_dir'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutters_dir'
    var_5 = get_config(var_0)[var_4]
    assert var_5 == '/custom/cookiecutters_dir'
    var_6 = 'invalid_yaml.yaml'
    var_7 = 'invalid: yaml: file'
    var_8 = module_0.get_config(var_6)
    var_9 = 'non_existent.yaml'
    var_10 = module_0.get_config(var_9)
    var_11 = 'non_dict.yaml'
    var_12 = 'not a dict'
    var_13 = module_0.get_config(var_11)
    var_14 = 'env_var_config.yaml'
    var_15 = 'cookiecutters_dir'
    var_16 = '$HOME/custom_cookiecutters'
    var_17 = {var_15: var_16}
    var_18 = get_config(var_14)[var_15]
    var_19 = '~/custom_cookiecutters'
    var_20 = 'nested_dict.yaml'
    var_21 = 'default_context'
    var_22 = 'key1'
    var_23 = 'key2'
    var_24 = 'value1'
    var_25 = 'value2'
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = {var_21: var_26}
    var_28 = 'default_context'
    var_29 = get_config(var_20)[var_28]



# Parsed testcases at query #12
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'cookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/'
    var_2 = module_0.get_config(var_0)
    var_3 = 'nonexistent.yaml'
    var_4 = module_0.get_config(var_3)
    var_5 = 'invalid.yaml'
    var_6 = 'invalid: yaml: file'
    var_7 = module_0.get_config(var_5)
    var_8 = 'invalid_top_level.yaml'
    var_9 = '- item1\n- item2'
    var_10 = module_0.get_config(var_8)



# Parsed testcases at query #13
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'replay_dir'
    var_7 = '/mock/path'
    var_8 = '/mock/replay'
    var_9 = {var_2: var_7, var_6: var_8}
    var_10 = module_0.get_user_config(var_7)
    var_11 = '/env/path'
    var_12 = '/env/replay'
    var_13 = {var_2: var_11, var_6: var_12}
    var_14 = module_0.get_user_config()
    var_15 = 'COOKIECUTTER_CONFIG'
    var_16 = module_0.get_user_config()
    var_17 = '/non/existent/path'
    var_18 = module_0.get_user_config(var_17)
    var_19 = '/invalid/yaml/path'
    var_20 = module_0.get_user_config(var_19)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = module_0.get_user_config()
    var_7 = '/env/config/path'
    var_8 = module_0.get_config(var_7)
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = module_0.get_user_config()
    var_11 = module_0.get_user_config()
    var_12 = '/custom/config/path'
    var_13 = module_0.get_user_config(var_12)
    var_14 = module_0.get_config(var_12)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '/custom/cookiecutters/'
    var_5 = '/custom/replay/'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'valid_config.yml'
    var_14 = module_0.get_config(var_13)
    var_15 = {var_9: var_10}
    var_16 = 'non_existent_config.yml'
    var_17 = module_0.get_config(var_16)
    var_18 = 'invalid yaml content'
    var_19 = 'invalid_config.yml'
    var_20 = module_0.get_config(var_19)
    var_21 = 'invalid_config.yml'
    var_22 = '- item1\n- item2'
    var_23 = 'not_dict_config.yml'
    var_24 = module_0.get_config(var_23)
    var_25 = 'not_dict_config.yml'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #17
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'config.yaml'
    var_2 = var_0 / var_1
    var_3 = 'cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\ndefault_context:\n  foo: bar\nabbreviations:\n  gh: https://github.com/{0}.git\n'
    var_4 = module_0.get_config(var_2)
    var_5 = 'nonexistent.yaml'
    var_6 = var_0 / var_5
    var_7 = module_0.get_config(var_6)
    var_8 = 'invalid.yaml'
    var_9 = 'invalid: yaml: file'
    var_10 = 'not_dict.yaml'
    var_11 = 'not a dict'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'
    var_6 = '/nonexistent/path'
    var_7 = 'invalid yaml'
    var_8 = '- item1\n'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = module_1.dump(var_4)
    var_7 = module_0.get_user_config()
    var_8 = 'COOKIECUTTER_CONFIG'
    var_9 = module_1.dump(var_4)
    var_10 = module_0.get_user_config()
    var_11 = module_0.get_user_config()



# Parsed testcases at query #20
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/test-config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = 'tests/nonexistent-config.yaml'
    var_3 = module_0.get_config(var_2)
    var_4 = 'tests/invalid-config.yaml'
    var_5 = module_0.get_config(var_4)
    var_6 = 'tests/invalid-top-level-config.yaml'
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    cookiecutters_dir: ~/custom_cookiecutters/\n    replay_dir: ~/custom_replay/\n    default_context:\n        key: value\n    abbreviations:\n        custom: https://custom.com/{0}.git\n    '
    var_1 = '~/custom_cookiecutters/'
    var_2 = '~/custom_replay/'
    var_3 = 'invalid: yaml: content'
    var_4 = 'not_a_dict'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'
    var_6 = module_0.get_config(var_0)
    var_7 = '/nonexistent/path'
    var_8 = 'invalid yaml content'
    var_9 = module_0.get_config(var_8)
    var_10 = '- item1\n'
    var_11 = '- item2\n'
    var_12 = module_0.get_config(var_10)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/test_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = 'tests/invalid_config.yaml'
    var_3 = module_0.get_config(var_2)
    var_4 = 'tests/non_existent_config.yaml'
    var_5 = module_0.get_config(var_4)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay'
    var_1 = '/nonexistent/path'
    var_2 = module_0.get_config(var_1)
    var_3 = 'invalid: yaml: here'
    var_4 = module_0.get_config(var_3)
    var_5 = '- list\n- items'
    var_6 = module_0.get_config(var_5)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '/tmp/cookiecutters'
    var_5 = '/tmp/replay'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'test'
    var_10 = 'https://example.com/{0}.git'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = '/path/to/nonexistent/file'
    var_14 = module_0.get_config(var_13)
    var_15 = 'invalid yaml content'
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/custom_cookiecutters'
    var_6 = '~/custom_replay'
    var_7 = 'key1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = {var_10: var_11}



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_1)
    var_3 = '/nonexistent/path'
    var_4 = module_0.get_config(var_3)
    var_5 = '- list\n- items'
    var_6 = module_0.get_config(var_5)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'valid_config.yaml'
    var_2 = 'cookiecutters_dir: /tmp/custom_cookiecutters\n'
    var_3 = 'replay_dir: /tmp/custom_replay\n'
    var_4 = 'default_context:\n'
    var_5 = '  key1: value1\n'
    var_6 = 'abbreviations:\n'
    var_7 = '  gh: "https://github.com/{0}.git"\n'
    var_8 = 'invalid_yaml.yaml'
    var_9 = 'invalid: yaml: here'
    var_10 = 'non_dict.yaml'
    var_11 = '- item1\n- item2'
    var_12 = 'nonexistent.yaml'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay'
    var_3 = 'invalid: yaml: here'
    var_4 = '- list\n- item'
    var_5 = 'nonexistent.yml'
    var_6 = var_4 / var_5
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/cookiecutter_replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'
    var_6 = '  gl: https://gitlab.com/{0}.git\n'
    var_7 = '  bb: https://bitbucket.org/{0}\n'
    var_8 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_9 = 'replay_dir: /tmp/cookiecutter_replay\n'
    var_10 = 'default_context:\n'
    var_11 = '  key1: value1\n'
    var_12 = 'abbreviations:\n'
    var_13 = '  gh: https://github.com/{0}.git\n'
    var_14 = '  gl: https://gitlab.com/{0}.git\n'
    var_15 = '  bb: https://bitbucket.org/{0}\n'
    var_16 = 'invalid_key: invalid_value\n'
    var_17 = '/tmp/non_existent_file'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '/tmp/cookiecutters'
    var_6 = '/tmp/replay'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'gh'
    var_11 = 'https://github.com/{0}.git'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = '/invalid/path/to/config.yml'
    var_15 = module_0.get_config(var_14)
    var_16 = 'invalid yaml'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/fake/config.yaml'
    var_7 = 'replay_dir'
    var_8 = '/custom/replay'
    var_9 = {var_7: var_8}
    var_10 = module_0.get_user_config(var_6)
    var_11 = 'default_context'
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = module_0.get_user_config()
    var_17 = 'abbreviations'
    var_18 = 'custom'
    var_19 = 'https://custom.com/{0}'
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = module_0.get_user_config()
    var_23 = 'COOKIECUTTER_CONFIG'
    var_24 = module_0.get_user_config()



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key: value\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = 'invalid: yaml: file\n'
    var_11 = '/tmp/non_existent_file'
    var_12 = '- item1\n'
    var_13 = '- item2\n'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /custom/path\n'
    var_1 = 'replay_dir: /custom/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://custom.github.com/{0}.git\n'
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_config(var_6)
    var_8 = 'invalid: yaml: file'
    var_9 = '- list\n- item'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir'
    var_7 = '/tmp/custom'
    var_8 = {var_6: var_7}
    var_9 = module_1.dump(var_8)
    var_10 = 'cookiecutters_dir'
    var_11 = '/tmp/env'
    var_12 = {var_10: var_11}
    var_13 = module_1.dump(var_12)
    var_14 = module_0.get_user_config()
    var_15 = 'COOKIECUTTER_CONFIG'
    var_16 = '/nonexistent/path'
    var_17 = module_0.get_user_config(var_16)
    var_18 = 'invalid: yaml: here'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/cookiecutter_replay\n'
    var_2 = '/non/existent/path'
    var_3 = module_0.get_config(var_2)
    var_4 = 'Expected ConfigDoesNotExistException'
    var_5 = AssertionError(var_4)
    var_6 = 'invalid yaml'
    var_7 = 'Expected InvalidConfiguration'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #16
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = var_0 / var_1
    var_3 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_4 = 'replay_dir: /tmp/replay\n'
    var_5 = 'default_context:\n'
    var_6 = '  key1: value1\n'
    var_7 = '  key2: value2\n'
    var_8 = 'abbreviations:\n'
    var_9 = '  gh: https://github.com/{0}.git\n'
    var_10 = module_0.get_config(var_2)
    var_11 = 'nonexistent.yml'
    var_12 = var_3 / var_11
    var_13 = module_0.get_config(var_12)
    var_14 = 'invalid_config.yml'
    var_15 = var_12 / var_14
    var_16 = 'invalid: yaml: file'
    var_17 = module_0.get_config(var_15)
    var_18 = 'invalid_config2.yml'
    var_19 = var_5 / var_18
    var_20 = '- item1\n- item2\n'
    var_21 = module_0.get_config(var_19)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir: /custom/path\n'
    var_2 = 'replay_dir: /another/path\n'
    var_3 = 'default_context:\n  key: value\n'
    var_4 = 'abbreviations:\n  gh: custom_gh_url\n'
    var_5 = '/nonexistent/path'
    var_6 = module_0.get_config(var_5)
    var_7 = 'invalid: yaml: here'
    var_8 = module_0.get_config(var_7)
    var_9 = '- list\n- item'
    var_10 = module_0.get_config(var_9)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '/custom/cookiecutters'
    var_6 = '/custom/replay'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}.git'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'valid_config.yml'
    var_15 = module_0.get_config(var_14)
    var_16 = 'invalid: yaml: file'
    var_17 = 'invalid_config.yml'
    var_18 = module_0.get_config(var_17)
    var_19 = 'nonexistent_config.yml'
    var_20 = module_0.get_config(var_19)
    var_21 = 'invalid_config.yml'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/cookiecutter_replay/\n        default_context:\n            project_name: Test Project\n        abbreviations:\n            gh: https://github.com/{0}.git\n        '
    var_1 = '/nonexistent/path'
    var_2 = module_0.get_config(var_1)
    var_3 = 'Expected ConfigDoesNotExistException'
    var_4 = AssertionError(var_3)
    var_5 = 'invalid: yaml: content'
    var_6 = 'Expected InvalidConfiguration'
    var_7 = AssertionError(var_6)
    var_8 = 'invalid_top_level'
    var_9 = 'Expected InvalidConfiguration'
    var_10 = AssertionError(var_9)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_2 = 'replay_dir: /custom/replay\n'
    var_3 = 'invalid: yaml: here'
    var_4 = '/nonexistent/path'
    var_5 = module_0.get_config(var_4)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = '~/custom_cookiecutters'
    var_4 = '~/custom_replay'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'invalid_config.yaml'
    var_7 = 'invalid: yaml: file'
    var_8 = 'non_existent.yaml'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay'
    var_3 = 'invalid: yaml: here'
    var_4 = '- item1\n- item2'
    var_5 = 'nonexistent.yml'



# Parsed testcases at query #23
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = module_0.get_user_config()



# Parsed testcases at query #24
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/test-configs/valid_config.yml'
    var_1 = module_0.get_config(var_0)
    var_2 = 'tests/test-configs/nonexistent_config.yml'
    var_3 = module_0.get_config(var_2)
    var_4 = 'tests/test-configs/invalid_yaml_config.yml'
    var_5 = module_0.get_config(var_4)
    var_6 = 'tests/test-configs/invalid_top_level_config.yml'
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_2 = 'replay_dir: /tmp/replay\n'
    var_3 = 'default_context:\n'
    var_4 = '  project_name: My Project\n'
    var_5 = 'abbreviations:\n'
    var_6 = '  gh: https://github.com/{0}.git\n'
    var_7 = 'invalid yaml'
    var_8 = '/tmp/non_existing_file'
    var_9 = '- item1\n'
    var_10 = '- item2\n'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/files/config_test_valid.yml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/.cookiecutters/'
    var_6 = '~/.cookiecutter_replay/'
    var_7 = []
    var_8 = module_0.get_config(var_0)
    var_9 = 'tests/files/config_test_invalid.yml'
    var_10 = module_0.get_config(var_9)
    var_11 = 'tests/files/config_test_invalid_yaml.yml'
    var_12 = module_0.get_config(var_11)
    var_13 = 'tests/files/config_test_not_dict.yml'
    var_14 = module_0.get_config(var_13)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = '  key2: value2\n'
    var_5 = 'abbreviations:\n'
    var_6 = '  gh: https://github.com/{0}.git\n'
    var_7 = '/non/existent/path'
    var_8 = module_0.get_config(var_7)
    var_9 = 'invalid: yaml: file\n'
    var_10 = 'not a dict\n'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = 'abbreviations'
    var_6 = '/custom/cookiecutters'
    var_7 = '/custom/replay'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'custom'
    var_12 = 'https://custom.com/{0}'
    var_13 = {var_11: var_12}
    var_14 = {var_2: var_6, var_3: var_7, var_4: var_10, var_5: var_13}
    var_15 = 'invalid: yaml: file'
    var_16 = '- item1\n- item2'
    var_17 = 'nonexistent.yml'
    var_18 = module_0.get_config(var_1)



# Parsed testcases at query #29
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_dir'
    var_6 = '~/test_replay'
    var_7 = {}
    var_8 = 'gh'
    var_9 = 'https://github.com/{0}.git'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_10}
    var_12 = module_0.get_config(var_0)



# Parsed testcases at query #30
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = 'invalid: yaml: file'
    var_9 = module_0.get_user_config(var_8)
    var_10 = 'cookiecutters_dir: /custom/path'
    var_11 = module_0.get_user_config(var_3)
    var_12 = module_0.get_user_config()
    var_13 = 'COOKIECUTTER_CONFIG'
    var_14 = module_0.get_user_config()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key1: value1\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = None
    var_11 = module_0.get_user_config()
    var_12 = 'All tests passed!'
    var_13 = print(var_12)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = var_0 / var_1
    var_3 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n            '
    var_4 = module_0.get_config(var_2)
    var_5 = 'invalid_config.yml'
    var_6 = 'invalid: yaml: file'
    var_7 = 'non_dict_config.yml'
    var_8 = 'just a string'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir'
    var_7 = '/custom/path'
    var_8 = {var_6: var_7}
    var_9 = module_1.dump(var_8)
    var_10 = 'cookiecutters_dir'
    var_11 = '/env/path'
    var_12 = {var_10: var_11}
    var_13 = module_1.dump(var_12)
    var_14 = get_user_config()[var_12]
    assert var_14 == '/env/path'
    var_15 = 'COOKIECUTTER_CONFIG'
    var_16 = 'cookiecutters_dir'
    var_17 = '/user/path'
    var_18 = {var_16: var_17}
    var_19 = module_1.dump(var_18)
    var_20 = get_user_config()[var_18]
    assert var_20 == '/user/path'
    var_21 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay'
    var_1 = 'invalid: yaml: file'
    var_2 = module_0.get_config(var_1)
    var_3 = '/nonexistent/file.yaml'
    var_4 = module_0.get_config(var_3)
    var_5 = 'cookiecutters_dir: $HOME/cookies\nreplay_dir: ~/replay'
    var_6 = '$HOME/cookies'
    var_7 = '~/replay'
    var_8 = '- item1\n- item2'
    var_9 = module_0.get_config(var_8)
    var_10 = ''



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = '\n    cookiecutters_dir: /custom/cookiecutters/\n    replay_dir: /custom/replay/\n    default_context:\n        full_name: "Test User"\n    abbreviations:\n        gh: "https://github.com/{0}.git"\n    '
    var_2 = 'invalid: yaml: file'
    var_3 = module_0.get_config(var_0)
    var_4 = '/nonexistent/path/config.yaml'
    var_5 = module_0.get_config(var_4)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/cookiecutter_replay\n'
    var_2 = "default_context: {'key': 'value'}\n"
    var_3 = "abbreviations: {'gh': 'https://github.com/{0}.git'}\n"
    var_4 = 'invalid yaml'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /custom/path'
    var_1 = 'invalid yaml'
    var_2 = '/non/existent/path'
    var_3 = module_0.get_config(var_2)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    assert var_1 == '/tmp/custom'
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /tmp/custom\n'
    var_7 = 'cookiecutters_dir'
    var_8 = 'cookiecutters_dir: /tmp/env\n'
    var_9 = 'cookiecutters_dir'
    var_10 = get_user_config()[var_9]
    var_11 = '/tmp/env'
    var_12 = 'COOKIECUTTER_CONFIG'
    var_13 = 'cookiecutters_dir: /tmp/home\n'
    var_14 = 'cookiecutters_dir'
    var_15 = get_user_config()[var_14]
    var_16 = '/tmp/home'
    var_17 = module_0.get_user_config()



# Parsed testcases at query #10
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n        cookiecutters_dir: /custom/cookiecutters\n        replay_dir: /custom/replay\n        default_context:\n            project_name: TestProject\n        abbreviations:\n            gh: https://custom.github.com/{0}.git\n        '
    var_1 = '/non/existent/path'
    var_2 = module_0.get_config(var_1)
    var_3 = 'invalid: yaml: content'
    var_4 = 'not a dict'



# Parsed testcases at query #11
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay'
    var_1 = '/nonexistent/path'
    var_2 = module_0.get_config(var_1)
    var_3 = 'invalid: yaml: here'
    var_4 = module_0.get_config(var_3)
    var_5 = '- list\n- item'
    var_6 = module_0.get_config(var_5)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function.'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_2 = 'replay_dir: /tmp/replay\n'
    var_3 = []
    var_4 = 'invalid: yaml: file\n'
    var_5 = module_0.get_config(var_4)
    var_6 = '/nonexistent/file'
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #13
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir: /custom/path\n'
    var_2 = 'replay_dir: /another/path\n'
    var_3 = 'invalid: yaml: file\n'
    var_4 = module_0.get_config(var_3)
    var_5 = '/nonexistent/path'
    var_6 = module_0.get_config(var_5)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /custom/path\n'
    var_1 = 'replay_dir: /another/path\n'
    var_2 = 'default_context:\n  key1: value1\n'
    var_3 = 'abbreviations:\n  custom: https://example.com/{0}.git\n'
    var_4 = '/nonexistent/path'
    var_5 = module_0.get_config(var_4)
    var_6 = 'invalid: yaml: here'
    var_7 = module_0.get_config(var_6)
    var_8 = '- item1\n- item2'
    var_9 = module_0.get_config(var_8)
    var_10 = 'cookiecutters_dir: ~/custom\n'
    var_11 = 'replay_dir: $HOME/replay\n'
    var_12 = '~/custom'
    var_13 = '$HOME/replay'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\n'
    var_2 = '/path/to/nonexistent/config.yml'
    var_3 = module_0.get_config(var_2)
    var_4 = 'invalid.yml'
    var_5 = 'invalid yaml content'
    var_6 = 'invalid_top_level.yml'
    var_7 = '- not a dict'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'All tests passed.'
    var_7 = print(var_6)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '/custom/cookiecutters'
    var_6 = '/custom/replay'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}.git'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'valid_config.yml'
    var_15 = module_0.get_config(var_14)
    var_16 = 'invalid: yaml: file'
    var_17 = 'invalid_config.yml'
    var_18 = module_0.get_config(var_17)
    var_19 = 'nonexistent.yml'
    var_20 = module_0.get_config(var_19)
    var_21 = 'invalid_config.yml'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = 'invalid yaml: ['
    var_9 = module_0.get_user_config(var_8)
    var_10 = 'cookiecutters_dir: /custom/path\nreplay_dir: /custom/replay'
    var_11 = module_0.get_user_config(var_3)
    var_12 = 'cookiecutters_dir: /env/path'
    var_13 = module_0.get_user_config()
    var_14 = 'COOKIECUTTER_CONFIG'
    var_15 = False
    var_16 = module_0.get_user_config()



# Parsed testcases at query #19
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/custom_cookiecutters/'
    var_6 = '~/custom_replay/'
    var_7 = 'project_name'
    var_8 = 'Test Project'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}.git'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.get_config(var_0)
    var_15 = '~/custom_cookiecutters/'
    var_16 = '~/custom_replay/'
    var_17 = 'non_existent_config.yaml'
    var_18 = module_0.get_config(var_17)
    var_19 = 'invalid_config.yaml'
    var_20 = 'invalid yaml content'
    var_21 = module_0.get_config(var_19)
    var_22 = 'non_dict_config.yaml'
    var_23 = 'not a dict'
    var_24 = module_0.get_config(var_22)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n        abbreviations:\n            custom: https://custom.com/{0}.git\n        '
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/nonexistent/path'
    var_6 = module_0.get_config(var_2)
    var_7 = 'invalid: yaml: content'
    var_8 = '- item1\n- item2'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir: /custom/path\nreplay_dir: /another/path'
    var_2 = module_0.get_config(var_0)
    var_3 = 'invalid_config.yaml'
    var_4 = 'invalid: yaml: file'
    var_5 = module_0.get_config(var_3)
    var_6 = 'nonexistent.yaml'
    var_7 = module_0.get_config(var_6)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'w'
    var_11 = module_0.get_user_config()
    var_12 = module_0.get_user_config()
    var_13 = 'All tests passed!'
    var_14 = print(var_13)



# Parsed testcases at query #23
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'test_config.yml'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_2 = 'replay_dir: /tmp/cookiecutter_replay\n'
    var_3 = module_0.get_config(var_0)
    var_4 = 'invalid_config.yml'
    var_5 = 'invalid'
    var_6 = module_0.get_config(var_4)
    var_7 = 'test_config.yml'
    var_8 = 'invalid_config.yml'



# Parsed testcases at query #24
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_3 = 'replay_dir: /custom/replay\n'
    var_4 = 'default_context:\n'
    var_5 = '  key1: value1\n'
    var_6 = 'abbreviations:\n'
    var_7 = '  gh: https://github.com/{0}.git\n'
    var_8 = 'invalid: yaml: file'
    var_9 = 'nonexistent.yml'
    var_10 = var_8 / var_9
    var_11 = module_0.get_config(var_10)



# Parsed testcases at query #25
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: ~/custom_cookiecutters\nreplay_dir: ~/custom_replay'
    var_1 = '~/custom_cookiecutters'
    var_2 = '~/custom_replay'
    var_3 = 'invalid: yaml: file'
    var_4 = '/path/to/nonexistent/file'
    var_5 = module_0.get_config(var_4)
    var_6 = '- item1\n- item2'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'invalid yaml'
    var_3 = 'invalid'
    var_4 = '/tmp/nonexistent'
    var_5 = module_0.get_config(var_4)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = 'replay_dir'
    var_6 = '/env/path'
    var_7 = {var_5: var_6}
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'invalid: yaml: file'
    var_11 = '/non/existent/path'
    var_12 = module_0.get_user_config(var_11)



# Parsed testcases at query #28
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_2 = 'replay_dir: /tmp/replay\n'
    var_3 = 'invalid: YAML: : syntax\n'
    var_4 = '/non/existent/path'
    var_5 = module_0.get_config(var_4)
    var_6 = '- list\n'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_1 = 'replay_dir: /tmp/replay\n'
    var_2 = 'default_context:\n'
    var_3 = '  key: value\n'
    var_4 = 'abbreviations:\n'
    var_5 = '  gh: https://github.com/{0}.git\n'
    var_6 = '/non/existent/path'
    var_7 = module_0.get_config(var_6)
    var_8 = 'invalid: yaml: here\n'
    var_9 = '- item1\n'
    var_10 = '- item2\n'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test the get_config function.'
    var_1 = 'test_config.yml'
    var_2 = var_0 / var_1
    var_3 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_4 = 'replay_dir: /custom/replay\n'
    var_5 = 'default_context:\n'
    var_6 = '  key1: value1\n'
    var_7 = 'abbreviations:\n'
    var_8 = '  custom: https://custom.com/{0}.git\n'
    var_9 = module_0.get_config(var_2)
    var_10 = 'nonexistent.yml'
    var_11 = var_3 / var_10
    var_12 = module_0.get_config(var_11)
    var_13 = 'invalid.yml'
    var_14 = var_11 / var_13
    var_15 = 'invalid: yaml: file'
    var_16 = module_0.get_config(var_14)
    var_17 = 'non_dict.yml'
    var_18 = var_5 / var_17
    var_19 = '- item1\n- item2'
    var_20 = module_0.get_config(var_18)



