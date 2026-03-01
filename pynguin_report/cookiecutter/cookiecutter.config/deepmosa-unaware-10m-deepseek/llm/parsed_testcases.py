####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.get_user_config(default_config=var_0)
    var_2 = 'cookiecutters_dir'
    var_3 = 'abbreviations'
    var_4 = '/custom/cookiecutters/'
    var_5 = 'custom'
    var_6 = 'https://custom.com/{0}'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.get_user_config(default_config=var_8)
    var_10 = '\ncookiecutters_dir: /tmp/test_cookiecutters/\nreplay_dir: /tmp/test_replay/\ndefault_context:\n  key1: value1\nabbreviations:\n  test: https://test.com/{0}\n'
    var_11 = '/tmp/test_cookiecutters/'
    var_12 = '/tmp/test_replay/'
    var_13 = '\ncookiecutters_dir: /env/cookiecutters/\nreplay_dir: /env/replay/\n'
    var_14 = module_0.get_user_config()
    var_15 = '/env/cookiecutters/'
    var_16 = '/env/replay/'
    var_17 = 'COOKIECUTTER_CONFIG'
    var_18 = None
    var_19 = '.backup'
    var_20 = 'COOKIECUTTER_CONFIG'
    var_21 = module_0.get_user_config()
    var_22 = 'invalid: yaml: ['
    var_23 = 'just a string'
    var_24 = '/non/existent/path/config.yaml'
    var_25 = module_0.get_user_config(var_24)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_4 = 'invalid: yaml: ['
    var_5 = str(var_4)
    var_6 = '- item1\n- item2'
    var_7 = str(var_6)
    var_8 = ''
    var_9 = '~/.cookiecutters/'
    var_10 = '~/.cookiecutter_replay/'
    var_11 = '\ncookiecutters_dir: ~/custom_cookiecutters\nreplay_dir: $HOME/custom_replay\n'
    var_12 = '~/custom_cookiecutters'
    var_13 = '$HOME/custom_replay'
    var_14 = '\nreplay_dir: /partial/replay\n'
    var_15 = '~/.cookiecutters/'
    var_16 = '\nabbreviations:\n  custom1: https://custom1.com/{0}\n  gh: https://custom.github.com/{0}.git\n'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)
    var_2 = 'invalid: yaml: ['
    var_3 = '- item1\n- item2'
    var_4 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_5 = '/custom/cookiecutters'
    var_6 = '/custom/replay'
    var_7 = '\ncookiecutters_dir: ~/test_cookiecutters\nreplay_dir: $HOME/test_replay\n'
    var_8 = '~/test_cookiecutters'
    var_9 = '~/test_replay'
    var_10 = ''
    var_11 = '~/.cookiecutters/'
    var_12 = '~/.cookiecutter_replay/'
    var_13 = 'null'
    var_14 = '\nabbreviations:\n  new: https://new.com/{0}\n'
    var_15 = 'replay_dir: /test/path'
    var_16 = '/test/path'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)
    var_2 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_3 = '~/.cookiecutter_replay/'
    var_4 = 'cookiecutters_dir: $HOME/custom_cookiecutters\nreplay_dir: ~/custom_replay\ndefault_context:\n  author: Test Author\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_5 = '~/custom_cookiecutters'
    var_6 = '~/custom_replay'
    var_7 = 'invalid: yaml: :'
    var_8 = '- item1\n- item2'
    var_9 = 'just a string'
    var_10 = ''
    var_11 = '~/.cookiecutters/'
    var_12 = '~/.cookiecutter_replay/'
    var_13 = 'cookiecutters_dir: /test/path'
    var_14 = module_0.get_config(var_13)
    var_15 = 'abbreviations:\n  gl: https://gitlab.custom.com/{0}\n  new: https://new.com/{0}\n'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n  project_name: Test Project\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = '\ncookiecutters_dir: ~/test_cookiecutters/\nreplay_dir: $HOME/test_replay/\n'
    var_4 = '~/test_cookiecutters/'
    var_5 = '$HOME/test_replay/'
    var_6 = 'invalid: yaml: ['
    var_7 = '- item1\n- item2'
    var_8 = ''
    var_9 = '~/.cookiecutters/'
    var_10 = '~/.cookiecutter_replay/'
    var_11 = '\nabbreviations:\n  custom1: https://custom1.com/{0}\n  gh: https://github.com/custom/{0}.git\n'
    var_12 = 'cookiecutters_dir: /test/path/'
    var_13 = module_0.get_config(var_12)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = '\ncookiecutters_dir: $HOME/custom_cookiecutters\nreplay_dir: ~/custom_replay\n'
    var_4 = '~'
    var_5 = 'custom_cookiecutters'
    var_6 = 'custom_replay'
    var_7 = 'invalid: yaml: ['
    var_8 = '- item1\n- item2'
    var_9 = 'just a string'
    var_10 = ''
    var_11 = '~/.cookiecutters/'
    var_12 = '~/.cookiecutter_replay/'
    var_13 = 'replay_dir: /partial/replay'
    var_14 = '~/.cookiecutters/'
    var_15 = 'replay_dir: /path/object/test'
    var_16 = module_0.get_config(var_15)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  project_name: Test Project\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: $HOME/custom_cookiecutters\nreplay_dir: ~/custom_replay\n'
    var_3 = '$HOME/custom_cookiecutters'
    var_4 = '~/custom_replay'
    var_5 = 'invalid: yaml: ['
    var_6 = 'just a string'
    var_7 = ''
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = '\nreplay_dir: /only/replay\n'
    var_11 = '~/.cookiecutters/'
    var_12 = '\nabbreviations:\n  custom: https://custom.com/{0}\n  gh: https://github.com/custom/{0}.git\n'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_2 = 'invalid: yaml: ['
    var_3 = '- item1\n- item2'
    var_4 = ''
    var_5 = '~/.cookiecutters/'
    var_6 = '~/.cookiecutter_replay/'
    var_7 = '\ncookiecutters_dir: $HOME/custom_cookiecutters\nreplay_dir: ~/custom_replay\n'
    var_8 = '$HOME/custom_cookiecutters'
    var_9 = '~/custom_replay'
    var_10 = '\nreplay_dir: /partial/replay\n'
    var_11 = '~/.cookiecutters/'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  project_name: Test Project\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = 'invalid: yaml: ['
    var_4 = '- item1\n- item2'
    var_5 = ''
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = '\nreplay_dir: $HOME/custom_replay\ncookiecutters_dir: ~/custom_cookiecutters\n'
    var_9 = '~/custom_replay'
    var_10 = '~/custom_cookiecutters'
    var_11 = '\nabbreviations:\n  custom: https://custom.com/{0}\n  gh: https://github.custom.com/{0}.git\n'
    var_12 = 'replay_dir: /test/path'
    var_13 = module_0.get_config(var_12)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0.get_config(var_0)
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = '\ncookiecutters_dir: $HOME/test_cookiecutters\nreplay_dir: ~/test_replay\n'
    var_4 = '~/test_cookiecutters'
    var_5 = '~/test_replay'
    var_6 = 'invalid: yaml: content'
    var_7 = '- item1\n- item2'
    var_8 = ''
    var_9 = '~/.cookiecutters/'
    var_10 = '~/.cookiecutter_replay/'
    var_11 = 'cookiecutters_dir: /test/path'
    var_12 = module_0.get_config(var_11)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: "~/custom_cookiecutters"\nreplay_dir: "~/custom_replay"\ndefault_context:\n    author_name: "Test Author"\nabbreviations:\n    custom: "https://custom.com/{0}"\n'
    var_2 = '~/custom_cookiecutters'
    var_3 = '~/custom_replay'
    var_4 = 'invalid: yaml: ['
    var_5 = '- item1\n- item2'
    var_6 = ''
    var_7 = '~/.cookiecutters/'
    var_8 = '~/.cookiecutter_replay/'
    var_9 = '\ncookiecutters_dir: "$HOME/env_cookiecutters"\nreplay_dir: "~/user_replay"\n'
    var_10 = '$HOME/env_cookiecutters'
    var_11 = '~/user_replay'
    var_12 = '\nreplay_dir: "~/partial_replay"\n'
    var_13 = '~/.cookiecutters/'
    var_14 = '~/partial_replay'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = '\ncookiecutters_dir: "~/test_cookiecutters"\nreplay_dir: "$HOME/test_replay"\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: "https://custom.com/{0}"\n'
    var_4 = '~/test_cookiecutters'
    var_5 = '$HOME/test_replay'
    var_6 = 'invalid: yaml: ['
    var_7 = str(var_6)
    var_8 = '- item1\n- item2'
    var_9 = str(var_8)
    var_10 = 'just a string'
    var_11 = str(var_10)
    var_12 = ''
    var_13 = '~/.cookiecutters/'
    var_14 = '~/.cookiecutter_replay/'
    var_15 = 'null'
    var_16 = '~/.cookiecutters/'
    var_17 = '~/.cookiecutter_replay/'
    var_18 = '\ncookiecutters_dir: "~/test_path"\nreplay_dir: "~/test_replay_path"\n'
    var_19 = module_0.get_config(var_18)
    var_20 = '~/test_path'
    var_21 = '~/test_replay_path'
    var_22 = '\ndefault_context:\n  new_key: new_value\nabbreviations:\n  gl: "https://overridden.gitlab.com/{0}.git"\n  new_abbr: "https://new.com/{0}"\n'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = '/non/existent/path/config.yaml'
    var_1 = module_0.get_config(var_0)
    var_2 = str(var_0)
    var_3 = '\ncookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\ndefault_context:\n    key1: value1\n    key2: value2\nabbreviations:\n    custom: https://custom.com/{0}\n'
    var_4 = 'invalid: yaml: ['
    var_5 = str(var_4)
    var_6 = '- item1\n- item2'
    var_7 = str(var_6)
    var_8 = ''
    var_9 = '~/.cookiecutters/'
    var_10 = '~/.cookiecutter_replay/'
    var_11 = '\ncookiecutters_dir: ~/custom_cookiecutters/\nreplay_dir: $HOME/custom_replay/\n'
    var_12 = '~/custom_cookiecutters/'
    var_13 = '$HOME/custom_replay/'
    var_14 = '\nreplay_dir: /partial/replay/\n'
    var_15 = '~/.cookiecutters/'
    var_16 = '\ndefault_context:\n    project_name: "Test Project"\n    author: "Test Author"\n'
    var_17 = 'replay_dir: /test/path/'
    var_18 = module_0.get_config(var_17)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = 'invalid: yaml: :'
    var_2 = '- item1\n- item2'
    var_3 = ''
    var_4 = '~/.cookiecutters/'
    var_5 = '~/.cookiecutter_replay/'
    var_6 = 'cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_7 = 'cookiecutters_dir: $HOME/custom_cookiecutters\nreplay_dir: ~/custom_replay\n'
    var_8 = '$HOME/custom_cookiecutters'
    var_9 = '~/custom_replay'
    var_10 = 'replay_dir: /partial/replay\ndefault_context:\n  partial_key: partial_value\n'
    var_11 = '~/.cookiecutters/'
    var_12 = 'abbreviations:\n  custom1: https://custom1.com/{0}\n  gh: https://custom.github.com/{0}.git\n'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_2 = '\ncookiecutters_dir: ~/test_cookiecutters\nreplay_dir: $HOME/test_replay\n'
    var_3 = '$HOME/test_replay'
    var_4 = '~/test_cookiecutters'
    var_5 = 'invalid: yaml: ['
    var_6 = '- item1\n- item2'
    var_7 = ''
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = '\nabbreviations:\n  gh: https://custom.github.com/{0}.git\n  new: https://new.com/{0}\n'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = 'invalid: yaml: :'
    var_2 = '- item1\n- item2'
    var_3 = ''
    var_4 = '~/.cookiecutters/'
    var_5 = '~/.cookiecutter_replay/'
    var_6 = 'cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_7 = 'cookiecutters_dir: $HOME/test_cookiecutters\nreplay_dir: ~/test_replay\n'
    var_8 = '$HOME/test_cookiecutters'
    var_9 = '~/test_replay'
    var_10 = 'cookiecutters_dir: /partial/cookiecutters\ndefault_context:\n  project_name: Test\n'
    var_11 = '~/.cookiecutter_replay/'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\n  key2: value2\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_2 = '/custom/cookiecutters'
    var_3 = '/custom/replay'
    var_4 = '\ncookiecutters_dir: $HOME/test_cookiecutters\nreplay_dir: ~/test_replay\n'
    var_5 = '$HOME/test_cookiecutters'
    var_6 = '~/test_replay'
    var_7 = '\ninvalid: [yaml: syntax\n'
    var_8 = '- item1\n- item2\n'
    var_9 = 'just a string'
    var_10 = ''
    var_11 = '~/.cookiecutters/'
    var_12 = '~/.cookiecutter_replay/'
    var_13 = 'null'
    var_14 = '~/.cookiecutters/'
    var_15 = '~/.cookiecutter_replay/'
    var_16 = '\ncookiecutters_dir: /partial/path\n'
    var_17 = '/partial/path'
    var_18 = '~/.cookiecutter_replay/'
    var_19 = '\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_20 = 'abbreviations'
    var_21 = len(var_0)
    assert var_21 == 4



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'nonexistent.yaml'
    var_1 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  key1: value1\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_2 = 'invalid: yaml: ['
    var_3 = '- item1\n- item2'
    var_4 = '\nreplay_dir: $HOME/custom_replay\ncookiecutters_dir: ~/custom_cookiecutters\n'
    var_5 = '$HOME/custom_replay'
    var_6 = '~/custom_cookiecutters'
    var_7 = ''
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = '\ncookiecutters_dir: /partial/path\n'
    var_11 = '~/.cookiecutter_replay/'



