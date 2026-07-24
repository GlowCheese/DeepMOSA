####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- a list\n- not a dict'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = 'test_url'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = (var_7, var_8)
    var_15 = [var_14]
    var_16 = 'non_existent.yaml'
    var_17 = 'invalid.yaml'
    var_18 = 'invalid: yaml: content: ['
    var_19 = 'non_dict.yaml'
    var_20 = '- list_item1\n- list_item2'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_dir/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- not a dict'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_dir/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #8
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.dump(var_13)
    var_15 = (var_7, var_8)
    var_16 = [var_15]
    var_17 = 'non_existent.yaml'
    var_18 = 'invalid.yaml'
    var_19 = 'invalid: yaml: content: ['
    var_20 = 'non_dict.yaml'
    var_21 = '- list item'
    var_22 = 'env_var.yaml'
    var_23 = '$HOME/test_cookiecutters/'
    var_24 = '$USER/test_replay/'
    var_25 = {var_1: var_23, var_2: var_24}
    var_26 = module_0.dump(var_25)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #10
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = "cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'https://custom.com/{0}'"
    var_7 = 'replay_dir: /env/dir'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = '/non/existent/file.yaml'
    var_11 = module_0.get_user_config(var_10)
    var_12 = 'invalid: yaml: content: ['
    var_13 = module_0.get_user_config(var_12)



# Parsed testcases at query #11
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.dump(var_13)
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: [unclosed'
    var_18 = 'non_dict.yaml'
    var_19 = '- list item 1\n- list item 2'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n        abbreviations:\n            custom: https://custom.example.com/{0}\n        '
    var_2 = '\n        cookiecutters_dir: $HOME/test_cookiecutters/\n        replay_dir: ~/test_replay/\n        '
    var_3 = 'non_existent.yaml'
    var_4 = 'invalid.yaml'
    var_5 = 'invalid: yaml: content: [unclosed'
    var_6 = 'non_dict.yaml'
    var_7 = '- list item'



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
    var_6 = 'cookiecutters_dir'
    var_7 = '/test/path'
    var_8 = {var_6: var_7}
    var_9 = 'replay_dir'
    var_10 = '/env/path'
    var_11 = {var_9: var_10}
    var_12 = module_0.get_user_config()
    var_13 = 'COOKIECUTTER_CONFIG'
    var_14 = 'default_context'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = module_0.get_user_config()



# Parsed testcases at query #14
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = 'https://test.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.dump(var_13)
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'non_dict.yaml'
    var_19 = '- list item'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #16
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '~/test_dir/'
    var_5 = '~/test_replay/'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'config.yaml'
    var_14 = module_0.dump(var_12)
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: [unclosed'
    var_18 = 'non_dict.yaml'
    var_19 = 'just a string'
    var_20 = '$HOME/test_dir/'
    var_21 = '$USER/test_replay/'
    var_22 = {var_0: var_20, var_1: var_21}
    var_23 = 'env_config.yaml'
    var_24 = module_0.dump(var_22)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = "cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'test'"
    var_7 = 'replay_dir: /env/dir'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = "default_context:\n  key: 'value'"
    var_11 = module_0.get_user_config()
    var_12 = 'COOKIECUTTER_CONFIG'
    var_13 = module_0.get_user_config()



# Parsed testcases at query #22
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '~/.test_cookiecutters/'
    var_5 = '~/.test_replay/'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'test'
    var_10 = 'https://test.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'test_config.yaml'
    var_14 = module_0.get_config(var_13)
    var_15 = 'non_existent_config.yaml'
    var_16 = module_0.get_config(var_15)
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'invalid_config.yaml'
    var_19 = module_0.get_config(var_18)
    var_20 = '- list item'
    var_21 = 'non_dict_config.yaml'
    var_22 = module_0.get_config(var_21)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = 'test_url'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = (var_7, var_8)
    var_15 = [var_14]
    var_16 = 'non_existent.yaml'
    var_17 = 'invalid.yaml'
    var_18 = 'invalid: yaml: content: ['
    var_19 = 'non_dict.yaml'
    var_20 = '- list_item1\n- list_item2'



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir'
    var_7 = '/test/dir'
    var_8 = {var_6: var_7}
    var_9 = 'replay_dir'
    var_10 = '/env/dir'
    var_11 = {var_9: var_10}
    var_12 = module_0.get_user_config()
    var_13 = 'COOKIECUTTER_CONFIG'
    var_14 = 'default_context'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = module_0.get_user_config()
    var_20 = '/non/existent/file.yaml'
    var_21 = module_0.get_user_config(var_20)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/path'
    var_7 = 'cookiecutters_dir: /env/path'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'cookiecutters_dir: /user/path'
    var_11 = module_0.get_user_config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = 'https://test.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #10
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
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'cookiecutterrc'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n        abbreviations:\n            custom: https://custom.com/{0}\n    '
    var_2 = '\n        cookiecutters_dir: $HOME/test/\n        replay_dir: ~/test/\n    '
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent'
    var_6 = 'invalid_yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict_yaml'
    var_9 = '- list item'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'valid_config.yaml'
    var_1 = 'cookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\n'
    var_2 = module_0.get_config(var_0)
    var_3 = []
    var_4 = 'non_existent_config.yaml'
    var_5 = module_0.get_config(var_4)
    var_6 = 'invalid_yaml.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = module_0.get_config(var_6)
    var_9 = 'non_dict_yaml.yaml'
    var_10 = '- list item 1\n- list item 2'
    var_11 = module_0.get_config(var_9)
    var_12 = 'env_config.yaml'
    var_13 = 'cookiecutters_dir: $HOME/test/cookiecutters/\nreplay_dir: ~/test/replay/\n'
    var_14 = module_0.get_config(var_12)
    var_15 = '$HOME/test/cookiecutters/'
    var_16 = '~/test/replay/'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = (var_7, var_8)
    var_15 = [var_14]
    var_16 = 'non_existent.yaml'
    var_17 = 'invalid.yaml'
    var_18 = 'invalid: yaml: content: ['
    var_19 = 'non_dict.yaml'
    var_20 = '- list item'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /custom/dir\n'
    var_7 = 'cookiecutters_dir: /env/dir\n'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'cookiecutters_dir: /user/dir\n'
    var_11 = module_0.get_user_config()



# Parsed testcases at query #17
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'tests/data/valid_config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '~/.cookiecutters/'
    var_3 = '~/.cookiecutter_replay/'
    var_4 = []
    var_5 = 'tests/data/non_existent_config.yaml'
    var_6 = module_0.get_config(var_5)
    var_7 = 'tests/data/invalid_yaml.yaml'
    var_8 = module_0.get_config(var_7)
    var_9 = 'tests/data/non_dict_yaml.yaml'
    var_10 = module_0.get_config(var_9)
    var_11 = 'tests/data/config_with_env_vars.yaml'
    var_12 = module_0.get_config(var_11)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #19
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.dump(var_13)
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'non_dict.yaml'
    var_19 = '- list item'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/test_cookiecutters/'
    var_6 = '~/test_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



