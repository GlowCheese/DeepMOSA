####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n            key2: value2\n        abbreviations:\n            custom: https://custom.com/{0}\n        '
    var_2 = '\n        cookiecutters_dir: $TEST_DIR/cookiecutters/\n        replay_dir: ~/replay/\n        '
    var_3 = '~/replay/'
    var_4 = 'non_existent.yaml'
    var_5 = 'invalid: yaml: content: ['
    var_6 = 'not a dict'



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
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #3
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
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = module_0.get_config(var_2)
    var_14 = '~/.test_cookiecutters/'
    var_15 = '~/.test_replay/'
    var_16 = '/non/existent/path.yaml'
    var_17 = module_0.get_config(var_16)
    var_18 = 'invalid: yaml: content: ['
    var_19 = module_0.get_config(var_18)
    var_20 = '- not a dict'
    var_21 = module_0.get_config(var_20)



# Parsed testcases at query #4
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



# Parsed testcases at query #5
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
    var_10 = 'test_url'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = module_0.get_config(var_2)
    var_14 = '~/.test_cookiecutters/'
    var_15 = '~/.test_replay/'
    var_16 = '/non/existent/path.yaml'
    var_17 = module_0.get_config(var_16)
    var_18 = 'invalid: yaml: content: ['
    var_19 = module_0.get_config(var_18)
    var_20 = '- not a dict'
    var_21 = module_0.get_config(var_20)



# Parsed testcases at query #6
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
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test_cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_3 = '$HOME/test_cookiecutters/'
    var_4 = '~/test_replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = 'this is not a dict'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = '/nonexistent/config'
    var_10 = module_0.get_user_config(var_9)
    var_11 = '/invalid/config'
    var_12 = module_0.get_user_config(var_11)



# Parsed testcases at query #9
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
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()
    var_11 = '/invalid/config'
    var_12 = module_0.get_user_config(var_11)
    var_13 = '/list/config'
    var_14 = module_0.get_user_config(var_13)



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


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\n'
    var_7 = 'replay_dir: /env/dir\n'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'default_context:\n  key: value\n'
    var_11 = module_0.get_user_config()



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
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/.test_cookiecutters/'
    var_6 = '~/.test_replay/'
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



# Parsed testcases at query #15
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_1 = '/non/existent/path.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'invalid: yaml: content: [unclosed'
    var_4 = 'just a string'
    var_5 = '\ncookiecutters_dir: $HOME/test_cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_6 = '~/test_cookiecutters/'
    var_7 = '~/test_replay/'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/replay/\n'
    var_3 = '~'
    var_4 = '/test/'
    var_5 = '~/replay/'
    var_6 = 'non_existent.yaml'
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: [unclosed'
    var_9 = 'non_dict.yaml'
    var_10 = '- list item'



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
    var_18 = '- list item'



# Parsed testcases at query #18
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
    var_19 = '- list item'



# Parsed testcases at query #19
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
    var_9 = {var_6: var_7}
    var_10 = 'replay_dir'
    var_11 = '/env/path'
    var_12 = {var_10: var_11}
    var_13 = module_0.get_user_config()
    var_14 = {var_10: var_11}
    var_15 = 'COOKIECUTTER_CONFIG'
    var_16 = 'default_context'
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = module_0.get_user_config()
    var_22 = 'default_context'
    var_23 = 'key'
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = {var_22: var_25}
    var_27 = '/non/existent/path.yaml'
    var_28 = module_0.get_user_config(var_27)



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
    var_18 = '- list item'



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
    var_10 = 'test'
    var_11 = 'https://test.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- not a dict'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir'
    var_1 = 'replay_dir'
    var_2 = 'default_context'
    var_3 = 'abbreviations'
    var_4 = '~/test_cookiecutters/'
    var_5 = '~/test_replay/'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'valid_config.yaml'
    var_14 = module_0.get_config(var_13)
    var_15 = 'non_existent_config.yaml'
    var_16 = module_0.get_config(var_15)
    var_17 = 'invalid_config.yaml'
    var_18 = 'invalid: yaml: content: ['
    var_19 = module_0.get_config(var_17)
    var_20 = 'non_dict_config.yaml'
    var_21 = '- not a dict'
    var_22 = module_0.get_config(var_20)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test_cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_3 = '$HOME/test_cookiecutters/'
    var_4 = '~/test_replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



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
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/replay/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/replay/'
    var_5 = 'nonexistent.yaml'
    var_6 = module_0.get_config(var_1)
    var_7 = 'invalid yaml content'
    var_8 = '- list item'



# Parsed testcases at query #27
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
    var_18 = '- not a dict'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: ~/test_dir/\nreplay_dir: ~/test_replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '~/test_dir/'
    var_3 = '~/test_replay/'
    var_4 = 'non_existent.yaml'
    var_5 = 'invalid.yaml'
    var_6 = 'invalid: yaml: content: ['
    var_7 = 'non_dict.yaml'
    var_8 = '- list item'



# Parsed testcases at query #31
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



# Parsed testcases at query #32
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
    var_18 = '- not a dict'



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'cookiecutterrc'
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
    var_14 = 'non_existent_config'
    var_15 = 'invalid_yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict_yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #38
#--------------------------


import cookiecutter.config as module_0

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
    var_9 = 'test'
    var_10 = 'test_url'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'test_config.yaml'
    var_14 = module_0.get_config(var_13)
    var_15 = 'non_existent_config.yaml'
    var_16 = module_0.get_config(var_15)
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'invalid_config.yaml'
    var_19 = module_0.get_config(var_18)
    var_20 = 'invalid_config.yaml'
    var_21 = '- not a dict'
    var_22 = 'non_dict_config.yaml'
    var_23 = module_0.get_config(var_22)
    var_24 = 'non_dict_config.yaml'



# Parsed testcases at query #39
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



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = 'this is not a dict'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: ~/expanded/cookiecutters/\nreplay_dir: ~/expanded/replay/\n'
    var_3 = '~/expanded/cookiecutters/'
    var_4 = '~/expanded/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #42
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/custom/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/non/existent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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
    var_20 = '/nonexistent/path.yaml'
    var_21 = module_0.get_user_config(var_20)



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/custom_cookiecutters/'
    var_6 = '~/custom_replay/'
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



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n        abbreviations:\n            custom: https://custom.com/{0}\n        '
    var_2 = '\n        cookiecutters_dir: $HOME/test/\n        replay_dir: ~/replay/\n        '
    var_3 = '$HOME/test/'
    var_4 = '~/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #49
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\nreplay_dir: /test/replay'
    var_7 = 'cookiecutters_dir: /env/dir'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'cookiecutters_dir: /user/dir'
    var_11 = module_0.get_user_config()
    var_12 = module_0.get_user_config()



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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



# Parsed testcases at query #52
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
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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
    var_11 = 'https://custom.com/{0}.git'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = 'non_existent.yaml'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list item'



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test_cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_3 = '$HOME/test_cookiecutters/'
    var_4 = '~/test_replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #58
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\n'
    var_7 = 'replay_dir: /env/dir\n'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'default_context:\n  key: value\n'
    var_11 = module_0.get_user_config()



# Parsed testcases at query #59
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
    var_18 = '- not a dict'



# Parsed testcases at query #60
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
    var_7 = 'replay_dir: /env/replay\ncookiecutters_dir: /env/cookies'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = "default_context:\n  key: 'value'"
    var_11 = module_0.get_user_config()



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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



# Parsed testcases at query #63
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = '/nonexistent/config'
    var_9 = module_0.get_user_config(var_8)
    var_10 = module_0.get_user_config()
    var_11 = module_0.get_user_config()
    var_12 = module_0.get_user_config()
    var_13 = module_0.get_user_config()



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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
    var_18 = '- list item'



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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



# Parsed testcases at query #68
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #69
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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



# Parsed testcases at query #75
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



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
    var_18 = '- list item'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '~'
    var_4 = '/test/'
    var_5 = 'nonexistent.yaml'
    var_6 = module_0.get_config(var_1)
    var_7 = 'invalid yaml: ['
    var_8 = 'not a dict'



# Parsed testcases at query #7
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
    var_20 = '/nonexistent/path.yaml'
    var_21 = module_0.get_user_config(var_20)
    var_22 = 'invalid: yaml: content'
    var_23 = module_0.get_user_config(var_22)



# Parsed testcases at query #8
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
    var_7 = '/custom/dir'
    var_8 = {var_6: var_7}
    var_9 = {var_6: var_7}
    var_10 = 'cookiecutters_dir'
    var_11 = '/env/dir'
    var_12 = {var_10: var_11}
    var_13 = module_0.get_user_config()
    var_14 = {var_10: var_11}
    var_15 = 'COOKIECUTTER_CONFIG'
    var_16 = 'cookiecutters_dir'
    var_17 = '/default/dir'
    var_18 = {var_16: var_17}
    var_19 = module_0.get_user_config()
    var_20 = {var_16: var_17}



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
    var_18 = '- list item'



# Parsed testcases at query #10
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
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/replay/\n'
    var_3 = '~'
    var_4 = '/test/'
    var_5 = '~/replay/'
    var_6 = 'non_existent.yaml'
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: [unclosed'
    var_9 = 'non_dict.yaml'
    var_10 = '- list item'



# Parsed testcases at query #16
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
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'non_dict.yaml'
    var_19 = '- list item'



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
    var_18 = '- list item'



# Parsed testcases at query #18
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
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'non_dict.yaml'
    var_19 = '- list item'



# Parsed testcases at query #19
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'cookiecutterrc'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/custom_cookiecutters/'
    var_6 = '~/custom_replay/'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'custom'
    var_11 = 'https://custom.com/{0}'
    var_12 = {var_10: var_11}
    var_13 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_12}
    var_14 = module_0.dump(var_13)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'nonexistent_file.yaml'
    var_1 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'invalid_config.yaml'
    var_1 = 'invalid: yaml: content: [unclosed'

def test_case_0():
    var_0 = 'non_dict_config.yaml'
    var_1 = '- list item 1\n- list item 2'



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
    var_18 = '- list item'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test_dir\nreplay_dir: ~/test_replay\n'
    var_3 = '$HOME/test_dir'
    var_4 = '~/test_replay'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid yaml content'
    var_7 = 'not a dict'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid yaml content'
    var_7 = '- list item'
    var_8 = ''



# Parsed testcases at query #24
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
    var_16 = 'invalid: yaml: content: [unclosed'
    var_17 = 'non_dict.yaml'
    var_18 = '- list item 1\n- list item 2'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/test/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/test/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #26
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
    var_15 = 'invalid_yaml.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict_yaml.yaml'
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key1: value1\n        abbreviations:\n            custom: https://custom.com/{0}\n        '
    var_2 = '\n        cookiecutters_dir: $HOME/test_cookiecutters/\n        replay_dir: ~/test_replay/\n        '
    var_3 = '$HOME/test_cookiecutters/'
    var_4 = '~/test_replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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



# Parsed testcases at query #32
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
    var_9 = 'custom'
    var_10 = 'https://custom.com/{0}'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_4, var_1: var_5, var_2: var_8, var_3: var_11}
    var_13 = 'test_config.yaml'
    var_14 = module_0.get_config(var_13)
    var_15 = 'non_existent_config.yaml'
    var_16 = module_0.get_config(var_15)
    var_17 = 'invalid_config.yaml'
    var_18 = 'invalid: yaml: content: ['
    var_19 = module_0.get_config(var_17)
    var_20 = 'non_dict_config.yaml'
    var_21 = '- not a dict'
    var_22 = module_0.get_config(var_20)



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'cookiecutters_dir: /test/dir\nreplay_dir: /test/replay\nabbreviations:\n  test: test_value'
    var_1 = '/non/existent/path.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'invalid: yaml: content: ['
    var_4 = module_0.get_config(var_3)
    var_5 = '- not a dict'
    var_6 = module_0.get_config(var_5)
    var_7 = 'cookiecutters_dir: ~/test/dir\nreplay_dir: ~/test/replay'
    var_8 = '~/test/dir'
    var_9 = '~/test/replay'



# Parsed testcases at query #36
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/replay/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict.yaml'
    var_9 = '- list item 1\n- list item 2'



# Parsed testcases at query #38
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



# Parsed testcases at query #39
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



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\n        cookiecutters_dir: /custom/cookiecutters/\n        replay_dir: /custom/replay/\n        default_context:\n            key: value\n        abbreviations:\n            custom: https://custom.com/{0}\n        '
    var_2 = '\n        cookiecutters_dir: $HOME/test/cookiecutters/\n        replay_dir: ~/test/replay/\n        '
    var_3 = '$HOME/test/cookiecutters/'
    var_4 = '~/test/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #41
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



# Parsed testcases at query #42
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



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\n'
    var_7 = 'replay_dir: /test/replay\n'
    var_8 = 'default_context:\n'
    var_9 = '  key: value\n'
    var_10 = 'cookiecutters_dir: /env/dir\n'
    var_11 = module_0.get_user_config()
    var_12 = 'COOKIECUTTER_CONFIG'
    var_13 = '/non/existent/file.yaml'
    var_14 = module_0.get_user_config(var_13)
    var_15 = 'invalid: yaml: content\n'
    var_16 = '- list item\n'
    var_17 = module_0.get_user_config(var_15)



# Parsed testcases at query #45
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/test/config'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = module_0.get_user_config()



# Parsed testcases at query #46
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\nabbreviations:\n  custom: "test"'
    var_7 = 'replay_dir: /env/dir'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'default_context:\n  key: value'
    var_11 = module_0.get_user_config()



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = 'cookiecutters_dir'
    var_2 = 'replay_dir'
    var_3 = 'default_context'
    var_4 = 'abbreviations'
    var_5 = '~/custom_cookiecutters/'
    var_6 = '~/custom_replay/'
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



# Parsed testcases at query #48
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
    var_15 = 'non_existent.yaml'
    var_16 = 'invalid.yaml'
    var_17 = 'invalid: yaml: content: ['
    var_18 = 'non_dict.yaml'
    var_19 = '- list item'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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



# Parsed testcases at query #52
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
    var_18 = '- not a dict'



# Parsed testcases at query #53
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



# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test_cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_3 = '$HOME/test_cookiecutters/'
    var_4 = '~/test_replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item 1\n- list item 2'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/cookiecutters/\nreplay_dir: ~/test/replay/\n'
    var_3 = '$HOME/test/cookiecutters/'
    var_4 = '~/test/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: [unclosed'
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #57
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
    var_17 = 'invalid yaml: ['
    var_18 = 'invalid_config.yaml'
    var_19 = module_0.get_config(var_18)
    var_20 = '- list item'
    var_21 = 'non_dict_config.yaml'
    var_22 = module_0.get_config(var_21)



# Parsed testcases at query #58
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



# Parsed testcases at query #59
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



# Parsed testcases at query #60
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = 'cookiecutters_dir: /test/dir\n'
    var_7 = 'cookiecutters_dir: /env/dir\n'
    var_8 = module_0.get_user_config()
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = 'cookiecutters_dir: /user/dir\n'
    var_11 = module_0.get_user_config()
    var_12 = module_0.get_user_config()



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'test_config.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key: value\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/test/\nreplay_dir: ~/replay/\n'
    var_3 = '$HOME/test/'
    var_4 = '~/replay/'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'non_dict.yaml'
    var_9 = '- list item'



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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
    var_18 = '- list_item1\n- list_item2'



# Parsed testcases at query #66
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/dir'
    var_4 = {var_2: var_3}
    var_5 = module_0.get_user_config(default_config=var_4)
    var_6 = '/nonexistent/path'
    var_7 = module_0.get_user_config(var_6)
    var_8 = module_0.get_user_config()
    var_9 = module_0.get_user_config()
    var_10 = 'cookiecutters_dir'
    var_11 = '/test/dir'
    var_12 = {var_10: var_11}
    var_13 = 'replay_dir'
    var_14 = '/test/replay'
    var_15 = {var_13: var_14}
    var_16 = module_0.get_user_config()
    var_17 = 'COOKIECUTTER_CONFIG'



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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
    var_18 = '- not a dict'



# Parsed testcases at query #72
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



