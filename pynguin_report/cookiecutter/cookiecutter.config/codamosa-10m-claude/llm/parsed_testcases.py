####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'valid_config.yaml'
    var_4 = 'cookiecutters_dir: /custom/path\nreplay_dir: /replay/path\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_5 = 'empty_config.yaml'
    var_6 = ''
    var_7 = 'TEST_DIR'
    var_8 = '/test/dir'
    var_9 = 'config_with_env.yaml'
    var_10 = 'cookiecutters_dir: $TEST_DIR/cookies\n'
    var_11 = 'config_with_home.yaml'
    var_12 = 'replay_dir: ~/my_replay\n'
    var_13 = 'replay_dir'
    var_14 = '~'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- item1\n- item2\n'
    var_19 = 'partial_config.yaml'
    var_20 = 'default_context:\n  key1: value1\n'



# Parsed testcases at query #2
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'config.yaml'
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
    var_12 = 'https://example.com/{0}'
    var_13 = {var_11: var_12}
    var_14 = {var_2: var_6, var_3: var_7, var_4: var_10, var_5: var_13}
    var_15 = module_0.dump(var_14)
    var_16 = 'config2.yaml'
    var_17 = 'TEST_DIR'
    var_18 = '/test/path'
    var_19 = '$TEST_DIR/cookiecutters'
    var_20 = '~/replay'
    var_21 = {var_2: var_19, var_3: var_20}
    var_22 = module_0.dump(var_21)
    var_23 = 'nonexistent.yaml'
    var_24 = 'invalid.yaml'
    var_25 = '{ invalid yaml content: ['
    var_26 = 'non_dict.yaml'
    var_27 = '- item1\n- item2'
    var_28 = 'empty.yaml'
    var_29 = ''
    var_30 = 'partial.yaml'
    var_31 = 'default_context:\n  author: Test Author'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay\ndefault_context:\n  author: Test Author\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_5 = 'empty.yaml'
    var_6 = ''
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content:'
    var_9 = 'nondict.yaml'
    var_10 = '- item1\n- item2\n'
    var_11 = 'env_config.yaml'
    var_12 = 'cookiecutters_dir: $HOME/.cookiecutters\n'
    var_13 = 'tilde_config.yaml'
    var_14 = 'replay_dir: ~/custom_replay\n'
    var_15 = 'replay_dir'
    var_16 = '~'
    var_17 = 'minimal.yaml'
    var_18 = 'cookiecutters_dir: /custom/path\n'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: ~/custom_cookiecutters/\nreplay_dir: ~/custom_replay/\ndefault_context:\n    author_name: John Doe\nabbreviations:\n    gh: https://github.com/{0}.git\n'
    var_3 = '~'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises exception for nonexistent file.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when YAML is not a dict.'
    var_1 = 'list.yaml'
    var_2 = '- item1\n- item2\n'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file.'
    var_1 = 'empty.yaml'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config merges user config with defaults.'
    var_1 = 'partial.yaml'
    var_2 = 'default_context:\n    custom_key: custom_value\n'

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'CUSTOM_DIR'
    var_2 = '/custom/path'
    var_3 = 'envvar.yaml'
    var_4 = 'cookiecutters_dir: $CUSTOM_DIR/cookiecutters\n'

def test_case_0():
    var_0 = 'Test get_config can override builtin abbreviations.'
    var_1 = 'abbrev.yaml'
    var_2 = 'abbreviations:\n    gh: https://custom.github.com/{0}.git\n'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /custom/path\ndefault_context:\n  project_name: my_project\n'
    var_5 = 'empty_config.yaml'
    var_6 = ''
    var_7 = 'config_with_env.yaml'
    var_8 = 'TEST_DIR'
    var_9 = '/test/directory'
    var_10 = 'cookiecutters_dir: $TEST_DIR/cookies\n'
    var_11 = 'config_with_home.yaml'
    var_12 = 'cookiecutters_dir: ~/my_cookies\n'
    var_13 = 'cookiecutters_dir'
    var_14 = '~'
    var_15 = 'invalid.yaml'
    var_16 = 'invalid: yaml: content: ['
    var_17 = 'non_dict.yaml'
    var_18 = '- item1\n- item2\n'
    var_19 = 'nested_config.yaml'
    var_20 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_21 = 'replay_config.yaml'
    var_22 = 'replay_dir: ~/my_replay\n'
    var_23 = 'replay_dir'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay'
    var_5 = 'config_with_env.yaml'
    var_6 = 'cookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: ~/replay'
    var_7 = 'empty.yaml'
    var_8 = ''
    var_9 = '~/.cookiecutters/'
    var_10 = '~/.cookiecutter_replay/'
    var_11 = 'config_abbrev.yaml'
    var_12 = 'abbreviations:\n  custom: https://example.com/{0}.git'
    var_13 = 'invalid.yaml'
    var_14 = 'invalid: yaml: content: ['
    var_15 = 'non_dict.yaml'
    var_16 = '- item1\n- item2'
    var_17 = 'config_context.yaml'
    var_18 = 'default_context:\n  author_name: John Doe\n  project_name: My Project'
    var_19 = 'config_expand.yaml'
    var_20 = 'cookiecutters_dir: ~/custom_cookiecutters'
    var_21 = 'cookiecutters_dir'
    var_22 = '~'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = '\ncookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  gh: https://github.com/{0}.git\n'
    var_4 = 'invalid.yaml'
    var_5 = 'invalid: yaml: content: ['
    var_6 = 'non_dict.yaml'
    var_7 = '- item1\n- item2\n'
    var_8 = 'empty.yaml'
    var_9 = ''
    var_10 = 'env_var_config.yaml'
    var_11 = 'TEST_HOME'
    var_12 = '\ncookiecutters_dir: $TEST_HOME/cookiecutters\nreplay_dir: ~/replay\n'
    var_13 = 'replay_dir'
    var_14 = '~'
    var_15 = 'partial_config.yaml'
    var_16 = 'default_context:\n  custom_key: custom_value\n'
    var_17 = 'nested_abbrev.yaml'
    var_18 = '\nabbreviations:\n  gh: https://custom-github.com/{0}.git\n  custom: https://custom.com/{0}\n'



# Parsed testcases at query #8
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'config.yaml'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = '/tmp/cookiecutters'
    var_7 = '/tmp/replay'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_3: var_6, var_4: var_7, var_5: var_10}
    var_12 = module_0.dump(var_11)
    var_13 = 'config_env.yaml'
    var_14 = 'TEST_DIR'
    var_15 = '/test/path'
    var_16 = '$TEST_DIR/cookiecutters'
    var_17 = '~/replay'
    var_18 = {var_3: var_16, var_4: var_17}
    var_19 = module_0.dump(var_18)
    var_20 = '~'
    var_21 = 'config_abbrev.yaml'
    var_22 = 'abbreviations'
    var_23 = 'custom'
    var_24 = 'https://custom.com/{0}.git'
    var_25 = {var_23: var_24}
    var_26 = {var_22: var_25}
    var_27 = module_0.dump(var_26)
    var_28 = 'config_invalid.yaml'
    var_29 = 'invalid: yaml: content: ['
    var_30 = 'config_non_dict.yaml'
    var_31 = '- item1\n- item2\n'
    var_32 = 'config_empty.yaml'
    var_33 = ''
    var_34 = '~/.cookiecutters/'
    var_35 = '~/.cookiecutter_replay/'
    var_36 = 'config_partial.yaml'
    var_37 = 'author'
    var_38 = 'John Doe'
    var_39 = {var_37: var_38}
    var_40 = {var_5: var_39}
    var_41 = module_0.dump(var_40)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    author_name: Test Author\nabbreviations:\n    custom: https://custom.com/{0}.git\n'
    var_3 = 'non_existent.yaml'
    var_4 = 'invalid.yaml'
    var_5 = 'invalid: yaml: content: ['
    var_6 = 'non_dict.yaml'
    var_7 = '- item1\n- item2'
    var_8 = 'empty.yaml'
    var_9 = ''
    var_10 = 'config_env.yaml'
    var_11 = 'cookiecutters_dir: $HOME/.cookiecutters'
    var_12 = 'cookiecutters_dir'
    var_13 = '~'
    var_14 = 'config_tilde.yaml'
    var_15 = 'replay_dir: ~/.cookiecutter_replay_custom'
    var_16 = 'replay_dir'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'valid_config.yaml'
    var_2 = '\ncookiecutters_dir: ~/my_cookiecutters/\nreplay_dir: ~/my_replay/\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_3 = 'utf-8'
    var_4 = '~'
    var_5 = 'non_existent.yaml'
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content: ['
    var_8 = 'invalid_structure.yaml'
    var_9 = '- item1\n- item2\n'
    var_10 = 'empty.yaml'
    var_11 = ''
    var_12 = 'env_config.yaml'
    var_13 = 'cookiecutters_dir: $HOME/.custom_cookiecutters/'
    var_14 = 'partial.yaml'
    var_15 = 'cookiecutters_dir: /custom/path/'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid YAML config file.'
    var_1 = 'test_config.yaml'
    var_2 = '\ncookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://custom.com/{0}.git\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises ConfigDoesNotExistException for non-existent file.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = str(var_2)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration for invalid YAML.'
    var_1 = 'invalid_config.yaml'
    var_2 = 'invalid: yaml: content: ['
    var_3 = module_0.get_config(var_0)

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration when top-level is not a dict.'
    var_1 = 'list_config.yaml'
    var_2 = '- item1\n- item2\n'
    var_3 = module_0.get_config(var_0)

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns default config.'
    var_1 = 'empty_config.yaml'
    var_2 = ''
    var_3 = '~/.cookiecutters/'
    var_4 = '~/.cookiecutter_replay/'

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'MY_CUSTOM_DIR'
    var_2 = '/custom/dir'
    var_3 = 'env_config.yaml'
    var_4 = '\ncookiecutters_dir: $MY_CUSTOM_DIR/cookiecutters\nreplay_dir: $MY_CUSTOM_DIR/replay\n'

def test_case_0():
    var_0 = 'Test get_config accepts Path objects.'
    var_1 = 'path_config.yaml'
    var_2 = 'cookiecutters_dir: ~/test\n'



# Parsed testcases at query #12
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = '\ncookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay\ndefault_context:\n    author_name: John Doe\nabbreviations:\n    custom: https://example.com/{0}.git\n'
    var_5 = '~/my_cookiecutters'
    var_6 = '~/my_replay'
    var_7 = 'minimal.yaml'
    var_8 = 'cookiecutters_dir: ~/custom_cookies\n'
    var_9 = '~/custom_cookies'
    var_10 = 'empty.yaml'
    var_11 = ''
    var_12 = 'invalid.yaml'
    var_13 = '{ invalid yaml: ['
    var_14 = 'non_dict.yaml'
    var_15 = '- item1\n- item2\n'
    var_16 = 'env_config.yaml'
    var_17 = 'cookiecutters_dir: $HOME/.my_cookies\n'
    var_18 = 'HOME'
    var_19 = '/home/testuser'
    var_20 = 'nested.yaml'
    var_21 = '\nabbreviations:\n    gh: https://github.com/custom/{0}.git\n    custom_bb: https://custom.bitbucket.org/{0}\n'
    var_22 = 'tilde.yaml'
    var_23 = 'replay_dir: ~/test_replay\n'
    var_24 = '~/test_replay'



# Parsed testcases at query #13
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = '~/.cookiecutters/'
    var_6 = '~/.cookiecutter_replay/'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = 'invalid: yaml: content: ['
    var_11 = 'not'
    var_12 = 'a'
    var_13 = 'dict'
    var_14 = [var_11, var_12, var_13]
    var_15 = ''
    var_16 = 'cookiecutters_dir'
    var_17 = 'replay_dir'
    var_18 = '$HOME/.cookiecutters/'
    var_19 = '${HOME}/.cookiecutter_replay/'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = '~/.cookiecutters/'
    var_22 = '~/.cookiecutter_replay/'
    var_23 = 'abbreviations'
    var_24 = 'custom'
    var_25 = 'https://custom.com/{0}.git'
    var_26 = {var_24: var_25}
    var_27 = {var_23: var_26}
    var_28 = 'cookiecutters_dir'
    var_29 = '~/.cookiecutters/'
    var_30 = {var_28: var_29}
    var_31 = module_0.get_config(var_28)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_4 = 'replay_dir: /custom/replay\n'
    var_5 = ''
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = 'cookiecutters_dir: $HOME/.custom_cookiecutters\n'
    var_9 = 'cookiecutters_dir'
    var_10 = '~'
    var_11 = 'cookiecutters_dir: ~/my_cookiecutters\n'
    var_12 = 'cookiecutters_dir'
    var_13 = '~'
    var_14 = 'invalid: yaml: content: ['
    var_15 = '- item1\n- item2\n'
    var_16 = 'abbreviations:\n'
    var_17 = '  custom: https://custom.com/{0}.git\n'
    var_18 = 'cookiecutters_dir: /test/path\n'
    var_19 = module_0.get_config(var_18)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/non/existent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\n'
    var_5 = 'config_env.yaml'
    var_6 = 'cookiecutters_dir: $HOME/.cookiecutters\n'
    var_7 = '~/.cookiecutters'
    var_8 = 'config_tilde.yaml'
    var_9 = 'replay_dir: ~/my_replay\n'
    var_10 = '~/my_replay'
    var_11 = 'empty_config.yaml'
    var_12 = ''
    var_13 = 'invalid_config.yaml'
    var_14 = 'invalid: yaml: content: ['
    var_15 = 'list_config.yaml'
    var_16 = '- item1\n- item2\n'
    var_17 = 'nested_config.yaml'
    var_18 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_19 = 'path_config.yaml'
    var_20 = 'cookiecutters_dir: /path/cookiecutters\n'
    var_21 = 'str_config.yaml'
    var_22 = 'replay_dir: /path/replay\n'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid config files.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_4 = 'empty_config.yaml'
    var_5 = ''
    var_6 = 'invalid.yaml'
    var_7 = 'invalid: yaml: content:'
    var_8 = 'non_dict.yaml'
    var_9 = '- item1\n- item2\n'
    var_10 = 'env_config.yaml'
    var_11 = 'cookiecutters_dir: $HOME/.cookiecutters\n'
    var_12 = 'cookiecutters_dir'
    var_13 = '~'
    var_14 = 'tilde_config.yaml'
    var_15 = 'replay_dir: ~/my_replay\n'
    var_16 = 'replay_dir'
    var_17 = 'context_config.yaml'
    var_18 = 'default_context:\n  project_name: my_project\n  author: John Doe\n'
    var_19 = 'partial_config.yaml'
    var_20 = 'cookiecutters_dir: /custom/path\n'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'nonexistent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = 'cookiecutters_dir: /custom/path\n'
    var_4 = 'env_config.yaml'
    var_5 = 'cookiecutters_dir: $HOME/.custom_cookiecutters\n'
    var_6 = 'home_config.yaml'
    var_7 = 'replay_dir: ~/.custom_replay\n'
    var_8 = 'abbrev_config.yaml'
    var_9 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_10 = 'invalid.yaml'
    var_11 = 'invalid: yaml: content: ]['
    var_12 = 'list_config.yaml'
    var_13 = '- item1\n- item2\n'
    var_14 = 'empty_config.yaml'
    var_15 = ''
    var_16 = 'context_config.yaml'
    var_17 = 'default_context:\n  project_name: myproject\n  author: John Doe\n'
    var_18 = 'multi_path_config.yaml'
    var_19 = 'cookiecutters_dir: /path1\nreplay_dir: /path2\nabbreviations:\n  custom1: https://custom1.com/{0}\n'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    author_name: Test Author\nabbreviations:\n    custom: https://example.com/{0}.git\n'
    var_4 = 'utf-8'
    var_5 = 'env_config.yaml'
    var_6 = 'TEST_DIR'
    var_7 = '\ncookiecutters_dir: $TEST_DIR/cookiecutters\nreplay_dir: ~/custom_replay\n'
    var_8 = 'invalid.yaml'
    var_9 = 'invalid: yaml: content: ['
    var_10 = 'invalid_dict.yaml'
    var_11 = '- item1\n- item2\n'
    var_12 = 'empty.yaml'
    var_13 = ''
    var_14 = 'partial.yaml'
    var_15 = 'cookiecutters_dir: /partial/path\n'
    var_16 = 'tilde_config.yaml'
    var_17 = 'cookiecutters_dir: ~/my_cookiecutters\n'
    var_18 = 'cookiecutters_dir'
    var_19 = '~'



# Parsed testcases at query #19
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay\n'
    var_5 = 'config2.yaml'
    var_6 = 'cookiecutters_dir: $HOME/.cookiecutters\n'
    var_7 = 'HOME'
    var_8 = '/home/testuser'
    var_9 = 'config3.yaml'
    var_10 = 'replay_dir: ~/my_replay\n'
    var_11 = 'replay_dir'
    var_12 = '~'
    var_13 = 'config4.yaml'
    var_14 = 'invalid: yaml: content: ['
    var_15 = 'config5.yaml'
    var_16 = '- item1\n- item2\n'
    var_17 = 'config6.yaml'
    var_18 = ''
    var_19 = 'config7.yaml'
    var_20 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_21 = 'config8.yaml'
    var_22 = 'default_context:\n  author_name: John Doe\n'
    var_23 = 'config9.yaml'
    var_24 = 'cookiecutters_dir: $HOME/test/~/.cookiecutters\n'
    var_25 = '/home/user'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'valid_config.yaml'
    var_2 = 'cookiecutters_dir: /tmp/test\nreplay_dir: /tmp/replay\ndefault_context:\n  author: Test Author\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_3 = 'env_config.yaml'
    var_4 = 'TEST_DIR'
    var_5 = '/expanded/path'
    var_6 = 'cookiecutters_dir: $TEST_DIR/cookies\n'
    var_7 = 'home_config.yaml'
    var_8 = 'cookiecutters_dir: ~/my_cookies\n'
    var_9 = 'cookiecutters_dir'
    var_10 = '~'
    var_11 = 'does_not_exist.yaml'
    var_12 = 'invalid.yaml'
    var_13 = 'invalid: yaml: content: [\n'
    var_14 = 'non_dict.yaml'
    var_15 = '- item1\n- item2\n'
    var_16 = 'empty.yaml'
    var_17 = ''
    var_18 = '~/.cookiecutters/'
    var_19 = '~/.cookiecutter_replay/'
    var_20 = 'partial.yaml'
    var_21 = 'default_context:\n  name: TestProject\n'



# Parsed testcases at query #21
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = 'cookiecutters_dir'
    var_4 = 'default_context'
    var_5 = '/custom/path'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.dump(var_9)
    var_11 = 'env_config.yaml'
    var_12 = 'TEST_DIR'
    var_13 = '/expanded/dir'
    var_14 = '$TEST_DIR/cookiecutters'
    var_15 = {var_3: var_14}
    var_16 = module_0.dump(var_15)
    var_17 = 'home_config.yaml'
    var_18 = 'replay_dir'
    var_19 = '~/my_replay'
    var_20 = {var_18: var_19}
    var_21 = module_0.dump(var_20)
    var_22 = '~'
    var_23 = 'invalid.yaml'
    var_24 = 'invalid: yaml: content: ['
    var_25 = 'non_dict.yaml'
    var_26 = '- item1\n- item2'
    var_27 = 'empty.yaml'
    var_28 = ''
    var_29 = 'abbrev_config.yaml'
    var_30 = 'abbreviations'
    var_31 = 'custom'
    var_32 = 'https://custom.com/{0}'
    var_33 = {var_31: var_32}
    var_34 = {var_30: var_33}
    var_35 = module_0.dump(var_34)
    var_36 = 'string_path.yaml'
    var_37 = '/test'
    var_38 = {var_3: var_37}
    var_39 = module_0.dump(var_38)
    var_40 = 'both_dirs.yaml'
    var_41 = '~/cookies'
    var_42 = '~/replay'
    var_43 = {var_3: var_41, var_18: var_42}
    var_44 = module_0.dump(var_43)



# Parsed testcases at query #22
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = 'abbreviations'
    var_7 = '~/my_cookiecutters'
    var_8 = '~/my_replay'
    var_9 = 'project_name'
    var_10 = 'my_project'
    var_11 = {var_9: var_10}
    var_12 = 'custom'
    var_13 = 'https://example.com/{0}'
    var_14 = {var_12: var_13}
    var_15 = {var_3: var_7, var_4: var_8, var_5: var_11, var_6: var_14}
    var_16 = module_0.dump(var_15)
    var_17 = 'config_with_env.yaml'
    var_18 = 'TEST_DIR'
    var_19 = '/test/path'
    var_20 = '$TEST_DIR/cookiecutters'
    var_21 = '$TEST_DIR/replay'
    var_22 = {var_3: var_20, var_4: var_21}
    var_23 = module_0.dump(var_22)
    var_24 = 'invalid.yaml'
    var_25 = '{ invalid yaml content: ['
    var_26 = 'non_dict.yaml'
    var_27 = '- item1\n- item2'
    var_28 = 'empty_config.yaml'
    var_29 = ''
    var_30 = 'partial_config.yaml'
    var_31 = '~/custom_dir'
    var_32 = {var_3: var_31}
    var_33 = module_0.dump(var_32)
    var_34 = 'tilde_config.yaml'
    var_35 = '~/.my_cookiecutters'
    var_36 = '~/.my_replay'
    var_37 = {var_3: var_35, var_4: var_36}
    var_38 = module_0.dump(var_37)



# Parsed testcases at query #23
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yaml'
    var_8 = '\ncookiecutters_dir: /tmp/custom_cookiecutters\nreplay_dir: /tmp/custom_replay\n'
    var_9 = 'env_config.yaml'
    var_10 = '\ncookiecutters_dir: /tmp/env_cookiecutters\n'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_15 = 'nonexistent.yaml'
    var_16 = module_0.get_user_config()
    var_17 = '.cookiecutterrc'
    var_18 = '\ncookiecutters_dir: /tmp/user_cookiecutters\n'
    var_19 = module_0.get_user_config()
    var_20 = '/nonexistent/path/config.yaml'
    var_21 = module_0.get_user_config(var_20)
    var_22 = 'invalid_config.yaml'
    var_23 = '{ invalid yaml content: ['
    var_24 = module_0.get_user_config(var_20)
    var_25 = 'config.yaml'
    var_26 = 'cookiecutters_dir: /should/be/ignored'
    var_27 = 'abbrev_config.yaml'
    var_28 = '\nabbreviations:\n  custom: https://custom.com/{0}\n'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'test_config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    author_name: Test Author\nabbreviations:\n    custom: https://custom.com/{0}.git\n'
    var_3 = 'utf-8'
    var_4 = 'test_config2.yaml'
    var_5 = 'TEST_DIR'
    var_6 = '/test/path'
    var_7 = 'cookiecutters_dir: $TEST_DIR/cookiecutters'
    var_8 = 'test_config3.yaml'
    var_9 = 'cookiecutters_dir: ~/my_cookiecutters'
    var_10 = 'cookiecutters_dir'
    var_11 = '~'
    var_12 = 'non_existent.yaml'
    var_13 = 'invalid.yaml'
    var_14 = '{ invalid yaml: ['
    var_15 = 'non_dict.yaml'
    var_16 = '- item1\n- item2'
    var_17 = 'empty.yaml'
    var_18 = ''
    var_19 = 'partial.yaml'
    var_20 = 'default_context:\n    key1: value1'



# Parsed testcases at query #25
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yml'
    var_8 = 'cookiecutters_dir: /tmp/cookiecutters\n'
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = False
    var_11 = 'exists'
    var_12 = lambda x: var_10
    var_13 = module_0.get_user_config()
    var_14 = 'env_config.yml'
    var_15 = 'replay_dir: /tmp/replay\n'
    var_16 = lambda x: var_10
    var_17 = module_0.get_user_config()
    var_18 = 'user_config.yml'
    var_19 = 'default_context:\n  project_name: test\n'
    var_20 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_21 = module_0.get_user_config()
    var_22 = lambda x: var_1
    var_23 = 'invalid.yml'
    var_24 = '{ invalid yaml content: ['
    var_25 = module_0.get_user_config(var_0)
    var_26 = '/nonexistent/config.yml'
    var_27 = module_0.get_user_config(var_26)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    author_name: Test Author\nabbreviations:\n    custom: https://custom.com/{0}.git\n'

def test_case_0():
    var_0 = 'Test get_config raises exception for non-existent file.'
    var_1 = 'nonexistent.yaml'

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when YAML root is not a dict.'
    var_1 = 'list.yaml'
    var_2 = '- item1\n- item2\n'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file.'
    var_1 = 'empty.yaml'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'TEST_DIR'
    var_2 = '/test/path'
    var_3 = 'config.yaml'
    var_4 = '\ncookiecutters_dir: $TEST_DIR/cookiecutters\nreplay_dir: ~/replay\n'
    var_5 = 'replay_dir'
    var_6 = 'replay'

def test_case_0():
    var_0 = 'Test get_config merges custom config with defaults.'
    var_1 = 'config.yaml'
    var_2 = '\ndefault_context:\n    custom_key: custom_value\n'



# Parsed testcases at query #27
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yaml'
    var_8 = 'cookiecutters_dir: /my/custom/cookiecutters\n'
    var_9 = 'env_config.yaml'
    var_10 = 'replay_dir: /env/replay\n'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'HOME'
    var_15 = module_0.get_user_config()
    var_16 = '.cookiecutterrc'
    var_17 = 'default_context:\n  author: Test Author\n'
    var_18 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_19 = module_0.get_user_config()
    var_20 = '/nonexistent/config.yaml'
    var_21 = module_0.get_user_config(var_20)
    var_22 = 'invalid.yaml'
    var_23 = 'invalid: yaml: content: ]['
    var_24 = module_0.get_user_config(var_20)
    var_25 = 'non_dict.yaml'
    var_26 = '- item1\n- item2\n'
    var_27 = module_0.get_user_config(var_20)
    var_28 = 'config_vars.yaml'
    var_29 = 'cookiecutters_dir: $HOME/.cookiecutters\n'



# Parsed testcases at query #28
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = {}
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = '~/.cookiecutters/'
    var_11 = '~/.cookiecutter_replay/'
    var_12 = 'cookiecutters_dir'
    var_13 = '$HOME/.cookiecutters/'
    var_14 = {var_12: var_13}
    var_15 = 'invalid: yaml: content: ['
    var_16 = '- item1\n- item2\n'
    var_17 = ''
    var_18 = 'abbreviations'
    var_19 = 'custom'
    var_20 = 'https://custom.com/{0}'
    var_21 = {var_19: var_20}
    var_22 = {var_18: var_21}
    var_23 = 'default_context'
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = {var_23: var_26}
    var_28 = module_0.get_config(var_23)



# Parsed testcases at query #29
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'test_config.yaml'
    var_8 = 'replay_dir'
    var_9 = '/test/path'
    var_10 = '/test/replay'
    var_11 = {var_3: var_9, var_8: var_10}
    var_12 = module_1.dump(var_11)
    var_13 = 'env_config.yaml'
    var_14 = 'default_context'
    var_15 = 'name'
    var_16 = 'test'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = module_1.dump(var_18)
    var_20 = 'COOKIECUTTER_CONFIG'
    var_21 = module_0.get_user_config()
    var_22 = False
    var_23 = 'os.path.exists'
    var_24 = lambda x: var_22
    var_25 = module_0.get_user_config()
    var_26 = 'user_config.yaml'
    var_27 = 'abbreviations'
    var_28 = 'custom'
    var_29 = 'https://custom.com'
    var_30 = {var_28: var_29}
    var_31 = {var_27: var_30}
    var_32 = module_1.dump(var_31)
    var_33 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_34 = module_0.get_user_config()
    var_35 = '/nonexistent/path/config.yaml'
    var_36 = module_0.get_user_config(var_35)
    var_37 = '/nonexistent/env/config.yaml'
    var_38 = lambda x: var_22
    var_39 = module_0.get_user_config()



# Parsed testcases at query #30
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = '~/.cookiecutters/'
    var_6 = '~/.cookiecutter_replay/'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = 'cookiecutters_dir'
    var_11 = '$HOME/.cookiecutters/'
    var_12 = {var_10: var_11}
    var_13 = '$HOME/.cookiecutters/'
    var_14 = 'invalid: yaml: content: ['
    var_15 = '- item1\n- item2\n'
    var_16 = ''
    var_17 = 'abbreviations'
    var_18 = 'custom'
    var_19 = 'https://custom.com/{0}'
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = 'cookiecutters_dir'
    var_23 = '/tmp/test'
    var_24 = {var_22: var_23}
    var_25 = module_0.get_config(var_22)



# Parsed testcases at query #31
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/non/existent/path/.cookiecutterrc'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /custom/path\n'
    var_5 = 'empty.yaml'
    var_6 = ''
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: ['
    var_9 = 'non_dict.yaml'
    var_10 = '- item1\n- item2\n'
    var_11 = 'config_env.yaml'
    var_12 = 'cookiecutters_dir: $HOME/.custom_cookiecutters\n'
    var_13 = 'config_tilde.yaml'
    var_14 = 'replay_dir: ~/custom_replay\n'
    var_15 = 'config_abbrev.yaml'
    var_16 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_17 = 'config_context.yaml'
    var_18 = 'default_context:\n  author: John Doe\n'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yml'
    var_2 = 'valid_config.yml'
    var_3 = '\ncookiecutters_dir: /custom/path\nreplay_dir: /replay/path\ndefault_context:\n    author_name: John Doe\nabbreviations:\n    custom: https://example.com/{0}.git\n'
    var_4 = 'cookiecutters_dir'
    var_5 = 'custom/path'
    var_6 = 'replay_dir'
    var_7 = 'replay/path'
    var_8 = 'invalid.yml'
    var_9 = 'invalid: yaml: content: ['
    var_10 = 'non_dict.yml'
    var_11 = '- item1\n- item2'
    var_12 = 'empty.yml'
    var_13 = ''
    var_14 = 'env_config.yml'
    var_15 = '\ncookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: ~/replay\n'
    var_16 = 'partial.yml'
    var_17 = '\ncookiecutters_dir: /custom/cookiecutters\n'
    var_18 = 'custom/cookiecutters'
    var_19 = 'path_config.yml'
    var_20 = 'cookiecutters_dir: /test/path'
    var_21 = 'test/path'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'cookiecutterrc'
    var_2 = '\ncookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\ndefault_context:\n  full_name: "Test User"\nabbreviations:\n  gh: https://github.com/{0}.git\n'
    var_3 = 'utf-8'
    var_4 = '~/.cookiecutters/'

def test_case_0():
    var_0 = 'Test get_config raises exception when config file does not exist.'
    var_1 = 'nonexistent.yaml'

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content:'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'Test get_config raises exception when YAML root is not a dict.'
    var_1 = 'list.yaml'
    var_2 = '- item1\n- item2\n'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns default config.'
    var_1 = 'empty.yaml'
    var_2 = ''
    var_3 = 'utf-8'
    var_4 = '~/.cookiecutters/'
    var_5 = '~/.cookiecutter_replay/'

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: $HOME/.cookiecutters/\nreplay_dir: ~/test_replay/\n'
    var_3 = 'utf-8'
    var_4 = 'HOME'
    var_5 = 'cookiecutters_dir'

def test_case_0():
    var_0 = 'Test get_config merges custom config with defaults.'
    var_1 = 'partial.yaml'
    var_2 = '\ndefault_context:\n  custom_key: custom_value\n'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'Test get_config preserves builtin abbreviations when merging.'
    var_1 = 'config.yaml'
    var_2 = '\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = 'utf-8'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'cookiecutterrc'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  author_name: Test Author\nabbreviations:\n  custom: https://custom.com/{0}.git\n'

def test_case_0():
    var_0 = 'Test get_config raises exception for non-existent file.'
    var_1 = 'nonexistent.yml'

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'cookiecutterrc'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when top-level is not a dict.'
    var_1 = 'cookiecutterrc'
    var_2 = '- item1\n- item2'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file.'
    var_1 = 'cookiecutterrc'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config expands environment variables and user home.'
    var_1 = 'cookiecutterrc'
    var_2 = '\ncookiecutters_dir: $HOME/.cookiecutters_custom\nreplay_dir: ~/replay_custom\n'
    var_3 = 'HOME'
    var_4 = 'cookiecutters_dir'
    var_5 = '.cookiecutters_custom'
    var_6 = 'replay_dir'
    var_7 = 'replay_custom'

def test_case_0():
    var_0 = 'Test get_config merges partial config with defaults.'
    var_1 = 'cookiecutterrc'
    var_2 = '\ncookiecutters_dir: /custom/dir\n'

def test_case_0():
    var_0 = 'Test get_config merges abbreviations while preserving builtins.'
    var_1 = 'cookiecutterrc'
    var_2 = '\nabbreviations:\n  myrepo: https://myrepo.com/{0}.git\n'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    full_name: "Test User"\nabbreviations:\n    custom: "https://example.com/{0}"\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises exception for non-existent file.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when top-level is not a dict.'
    var_1 = 'list.yaml'
    var_2 = '- item1\n- item2'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns default config.'
    var_1 = 'empty.yaml'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config expands environment variables and home directory.'
    var_1 = 'TEST_VAR'
    var_2 = '/test/path'
    var_3 = 'config.yaml'
    var_4 = '\ncookiecutters_dir: $TEST_VAR/cookiecutters\nreplay_dir: ~/replay\n'
    var_5 = 'replay_dir'
    var_6 = 'replay'

def test_case_0():
    var_0 = 'Test get_config merges user config with defaults.'
    var_1 = 'config.yaml'
    var_2 = '\ndefault_context:\n    project_name: "My Project"\n'

def test_case_0():
    var_0 = 'Test get_config properly merges nested dictionaries.'
    var_1 = 'config.yaml'
    var_2 = '\nabbreviations:\n    custom: "https://custom.com/{0}"\n'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid YAML configuration file.'
    var_1 = 'test_config.yaml'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = 'abbreviations'
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'gh'
    var_12 = 'https://github.com/{0}.git'
    var_13 = {var_11: var_12}
    var_14 = {var_2: var_6, var_3: var_7, var_4: var_10, var_5: var_13}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = "Test get_config raises ConfigDoesNotExistException when file doesn't exist."
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration for invalid YAML.'
    var_1 = 'invalid_config.yaml'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration when top-level is not a dict.'
    var_1 = 'list_config.yaml'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'Test get_config handles empty YAML file.'
    var_1 = 'empty_config.yaml'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config expands environment variables and user paths.'
    var_1 = 'path_config.yaml'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = '~/custom_cookiecutters'
    var_5 = '~/.custom_replay'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '~'

def test_case_0():
    var_0 = 'Test get_config merges provided config with default values.'
    var_1 = 'partial_config.yaml'
    var_2 = 'default_context'
    var_3 = 'custom_key'
    var_4 = 'custom_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config accepts Path object.'
    var_1 = 'path_obj_config.yaml'
    var_2 = var_0 / var_1
    var_3 = 'default_context'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = module_0.get_config(var_2)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'COOKIECUTTER_CONFIG'
    var_8 = False
    var_9 = 'os.path.exists'
    var_10 = lambda x: var_8
    var_11 = module_0.get_user_config()
    var_12 = '.cookiecutterrc'
    var_13 = 'cookiecutters_dir: /tmp/test\nreplay_dir: /tmp/replay'
    var_14 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_15 = module_0.get_user_config()
    var_16 = 'custom.yaml'
    var_17 = 'cookiecutters_dir: /custom/test'
    var_18 = 'env_config.yaml'
    var_19 = 'cookiecutters_dir: /env/path'
    var_20 = '/nonexistent/path'
    var_21 = module_0.get_user_config()
    var_22 = '/nonexistent/config.yaml'
    var_23 = module_0.get_user_config()
    var_24 = 'invalid.yaml'
    var_25 = '{ invalid yaml: ['
    var_26 = module_0.get_user_config(var_23)
    var_27 = 'non_dict.yaml'
    var_28 = '- item1\n- item2'
    var_29 = module_0.get_user_config(var_23)
    var_30 = 'empty.yaml'
    var_31 = ''
    var_32 = module_0.get_user_config()
    var_33 = 'TEST_VAR'
    var_34 = '/expanded'
    var_35 = 'env_var.yaml'
    var_36 = 'cookiecutters_dir: $TEST_VAR/cookies'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid config files.'
    var_1 = 'non_existent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = '\ncookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\ndefault_context:\n  author_name: Test Author\nabbreviations:\n  gh: https://github.com/{0}.git\n'
    var_4 = 'config_with_env.yaml'
    var_5 = '\ncookiecutters_dir: $HOME/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\n'
    var_6 = '$HOME'
    var_7 = 'invalid.yaml'
    var_8 = '{ invalid yaml content: ['
    var_9 = 'non_dict.yaml'
    var_10 = '- item1\n- item2'
    var_11 = 'empty.yaml'
    var_12 = ''
    var_13 = 'partial_config.yaml'
    var_14 = 'default_context:\n  custom_key: custom_value'
    var_15 = 'config_tilde.yaml'
    var_16 = 'cookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay'
    var_17 = 'cookiecutters_dir'
    var_18 = '~'
    var_19 = 'replay_dir'
    var_20 = 'config_abbrev.yaml'
    var_21 = 'abbreviations:\n  custom: https://custom.com/{0}'



# Parsed testcases at query #4
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yml'
    var_8 = 'replay_dir'
    var_9 = '/custom/cookies'
    var_10 = '/custom/replay'
    var_11 = {var_3: var_9, var_8: var_10}
    var_12 = module_1.dump(var_11)
    var_13 = 'env_config.yml'
    var_14 = 'default_context'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = module_1.dump(var_18)
    var_20 = 'COOKIECUTTER_CONFIG'
    var_21 = False
    var_22 = module_0.get_user_config()
    var_23 = 'os.path.exists'
    var_24 = lambda x: var_21
    var_25 = module_0.get_user_config()
    var_26 = 'user_config.yml'
    var_27 = 'abbreviations'
    var_28 = 'custom'
    var_29 = 'https://custom.com/{0}'
    var_30 = {var_28: var_29}
    var_31 = {var_27: var_30}
    var_32 = module_1.dump(var_31)
    var_33 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_34 = module_0.get_user_config()
    var_35 = '/non/existent/path.yml'
    var_36 = module_0.get_user_config()
    var_37 = '/non/existent/config.yml'
    var_38 = module_0.get_user_config(var_37)
    var_39 = lambda x: var_38
    var_40 = '/priority/path'
    var_41 = {var_3: var_40}
    var_42 = module_0.get_user_config(default_config=var_41)
    var_43 = 'invalid.yml'
    var_44 = '{ invalid yaml content: ['
    var_45 = module_0.get_user_config(var_37)
    var_46 = 'non_dict.yml'
    var_47 = '- item1\n- item2'
    var_48 = module_0.get_user_config(var_37)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'valid_config.yaml'
    var_2 = 'cookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://example.com/{0}.git\n'
    var_3 = 'cookiecutters_dir'
    var_4 = 'my_cookiecutters'
    var_5 = 'replay_dir'
    var_6 = 'my_replay'
    var_7 = 'env_config.yaml'
    var_8 = 'cookiecutters_dir: $HOME/.cookiecutters\n'
    var_9 = 'empty_config.yaml'
    var_10 = ''
    var_11 = 'nonexistent.yaml'
    var_12 = 'invalid.yaml'
    var_13 = 'cookiecutters_dir: ~/my_cookiecutters\n  invalid: : : syntax\n'
    var_14 = 'non_dict.yaml'
    var_15 = '- item1\n- item2\n'
    var_16 = 'partial_config.yaml'
    var_17 = 'default_context:\n  project_name: My Project\n'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'valid_config.yaml'
    var_2 = '\ncookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\ndefault_context:\n    author_name: John Doe\nabbreviations:\n    gh: https://github.com/{0}.git\n'
    var_3 = 'config_with_env.yaml'
    var_4 = 'cookiecutters_dir: $HOME/.cookiecutters'
    var_5 = 'cookiecutters_dir'
    var_6 = '$HOME'
    var_7 = 'config_with_tilde.yaml'
    var_8 = 'replay_dir: ~/replay'
    var_9 = 'replay_dir'
    var_10 = '~'
    var_11 = 'empty_config.yaml'
    var_12 = ''
    var_13 = 'non_existent.yaml'
    var_14 = 'invalid.yaml'
    var_15 = '{ invalid yaml content: ['
    var_16 = 'non_dict.yaml'
    var_17 = '- item1\n- item2'
    var_18 = 'partial_config.yaml'
    var_19 = 'cookiecutters_dir: /custom/path'
    var_20 = 'nested_config.yaml'
    var_21 = '\nabbreviations:\n    custom: https://custom.com/{0}\n'



# Parsed testcases at query #7
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/non/existent/path/.cookiecutterrc'
    var_2 = module_0.get_config(var_1)
    var_3 = 'test_config.yml'
    var_4 = 'cookiecutters_dir'
    var_5 = 'replay_dir'
    var_6 = 'default_context'
    var_7 = '~/.custom_cookiecutters/'
    var_8 = '~/.custom_replay/'
    var_9 = 'author'
    var_10 = 'Test Author'
    var_11 = {var_9: var_10}
    var_12 = {var_4: var_7, var_5: var_8, var_6: var_11}
    var_13 = module_1.dump(var_12)
    var_14 = 'empty_config.yml'
    var_15 = ''
    var_16 = 'invalid_config.yml'
    var_17 = '{ invalid: yaml: content: ['
    var_18 = 'non_dict_config.yml'
    var_19 = '- item1\n- item2'
    var_20 = 'env_config.yml'
    var_21 = 'TEST_DIR'
    var_22 = '/test/directory'
    var_23 = '$TEST_DIR/cookiecutters/'
    var_24 = '$TEST_DIR/replay/'
    var_25 = {var_4: var_23, var_5: var_24}
    var_26 = module_1.dump(var_25)
    var_27 = 'abbrev_config.yml'
    var_28 = 'abbreviations'
    var_29 = 'custom'
    var_30 = 'https://custom.com/{0}.git'
    var_31 = {var_29: var_30}
    var_32 = {var_28: var_31}
    var_33 = module_1.dump(var_32)



# Parsed testcases at query #8
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yml'
    var_8 = "cookiecutters_dir: /tmp/custom\nabbreviations:\n  gh: 'https://github.com/{0}.git'"
    var_9 = 'env_config.yml'
    var_10 = 'cookiecutters_dir: /env/path'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_15 = '/nonexistent/path'
    var_16 = module_0.get_user_config()
    var_17 = '.cookiecutterrc'
    var_18 = 'cookiecutters_dir: /user/path'
    var_19 = module_0.get_user_config()
    var_20 = '/nonexistent/env/config.yml'
    var_21 = module_0.get_user_config()
    var_22 = 'replay_dir'
    var_23 = '/custom/replay'
    var_24 = {var_22: var_23}



# Parsed testcases at query #9
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /custom/path\n'
    var_5 = 'config_env.yaml'
    var_6 = 'cookiecutters_dir: $HOME/.custom_cookies\n'
    var_7 = 'HOME'
    var_8 = '/home/testuser'
    var_9 = 'config_tilde.yaml'
    var_10 = 'replay_dir: ~/.custom_replay\n'
    var_11 = 'config_invalid.yaml'
    var_12 = 'invalid: yaml: content: ['
    var_13 = 'config_list.yaml'
    var_14 = '- item1\n- item2\n'
    var_15 = 'config_empty.yaml'
    var_16 = ''
    var_17 = 'config_abbrev.yaml'
    var_18 = "abbreviations:\n  custom: 'https://custom.com/{0}.git'\n"
    var_19 = 'config_context.yaml'
    var_20 = 'default_context:\n  project_name: my_project\n  author: John Doe\n'
    var_21 = 'config_str.yaml'
    var_22 = 'cookiecutters_dir: /path/to/cookies\n'



# Parsed testcases at query #10
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yml'
    var_2 = 'valid_config.yml'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = '/tmp/cookies'
    var_7 = '/tmp/replay'
    var_8 = 'project_name'
    var_9 = 'my_project'
    var_10 = {var_8: var_9}
    var_11 = {var_3: var_6, var_4: var_7, var_5: var_10}
    var_12 = module_0.dump(var_11)
    var_13 = 'empty_config.yml'
    var_14 = ''
    var_15 = 'invalid.yml'
    var_16 = '{ invalid yaml: ['
    var_17 = 'non_dict.yml'
    var_18 = '- item1\n- item2'
    var_19 = 'env_config.yml'
    var_20 = 'TEST_DIR'
    var_21 = '$TEST_DIR/cookies'
    var_22 = '~/replay'
    var_23 = {var_3: var_21, var_4: var_22}
    var_24 = module_0.dump(var_23)
    var_25 = '~'
    var_26 = 'partial_config.yml'
    var_27 = 'abbreviations'
    var_28 = 'custom'
    var_29 = 'https://custom.com/{0}.git'
    var_30 = {var_28: var_29}
    var_31 = {var_27: var_30}
    var_32 = module_0.dump(var_31)
    var_33 = 'full_config.yml'
    var_34 = '/custom/cookies'
    var_35 = '/custom/replay'
    var_36 = 'author_name'
    var_37 = 'project_slug'
    var_38 = 'John Doe'
    var_39 = {var_36: var_38, var_37: var_9}
    var_40 = 'gh'
    var_41 = 'https://github.company.com/{0}.git'
    var_42 = {var_40: var_41}
    var_43 = {var_3: var_34, var_4: var_35, var_5: var_39, var_27: var_42}
    var_44 = module_0.dump(var_43)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = '\ncookiecutters_dir: ~/my_cookiecutters/\nreplay_dir: ~/my_replay/\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://custom.com/{0}.git\n'
    var_5 = 'empty.yaml'
    var_6 = ''
    var_7 = 'partial.yaml'
    var_8 = '\ndefault_context:\n  project_name: my_project\n'
    var_9 = 'invalid.yaml'
    var_10 = 'invalid: yaml: content:'
    var_11 = 'nondict.yaml'
    var_12 = '- item1\n- item2'
    var_13 = 'env.yaml'
    var_14 = 'TEST_DIR'
    var_15 = '\ncookiecutters_dir: $TEST_DIR/cookies/\nreplay_dir: ~/replay/\n'
    var_16 = 'abbrev.yaml'
    var_17 = '\nabbreviations:\n  gh: https://github.custom.com/{0}.git\n  new: https://newsite.com/{0}.git\n'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'nonexistent.yaml'
    var_2 = 'valid_config.yaml'
    var_3 = 'cookiecutters_dir: ~/custom_cookiecutters/\n'
    var_4 = 'cookiecutters_dir'
    var_5 = 'custom_cookiecutters/'
    var_6 = 'env_config.yaml'
    var_7 = 'TEST_VAR'
    var_8 = '/test/path'
    var_9 = 'cookiecutters_dir: $TEST_VAR/cookiecutters\n'
    var_10 = 'home_config.yaml'
    var_11 = 'replay_dir: ~/custom_replay/\n'
    var_12 = 'replay_dir'
    var_13 = 'custom_replay/'
    var_14 = 'invalid.yaml'
    var_15 = 'invalid: yaml: content: ['
    var_16 = 'non_dict.yaml'
    var_17 = '- item1\n- item2\n'
    var_18 = 'empty.yaml'
    var_19 = ''
    var_20 = 'abbrev_config.yaml'
    var_21 = "abbreviations:\n  custom: 'https://custom.com/{0}'\n"
    var_22 = 'context_config.yaml'
    var_23 = 'default_context:\n  author: John Doe\n  email: john@example.com\n'
    var_24 = 'string_path_config.yaml'
    var_25 = 'cookiecutters_dir: ~/test/\n'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\ndefault_context:\n  project_name: my_project\nabbreviations:\n  gh: https://github.com/{0}.git\n'

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: ~/replay\n'
    var_3 = 'cookiecutters_dir'
    var_4 = '/'
    var_5 = 'replay_dir'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises exception when file does not exist.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'config.yaml'
    var_2 = 'invalid: yaml: content:'

def test_case_0():
    var_0 = 'Test get_config raises exception when YAML is not a dict.'
    var_1 = 'config.yaml'
    var_2 = '- item1\n- item2'

def test_case_0():
    var_0 = 'Test get_config merges user config with default config.'
    var_1 = 'config.yaml'
    var_2 = '\ndefault_context:\n  custom_key: custom_value\n'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file.'
    var_1 = 'config.yaml'
    var_2 = ''

def test_case_0():
    var_0 = 'Test get_config preserves nested dictionary values.'
    var_1 = 'config.yaml'
    var_2 = '\nabbreviations:\n  custom: https://custom.com/{0}\n'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay\nabbreviations:\n  my_abbr: https://example.com/{0}.git\n'
    var_5 = '~/my_cookiecutters'
    var_6 = '~/my_replay'
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: ['
    var_9 = 'non_dict.yaml'
    var_10 = '- item1\n- item2\n'
    var_11 = 'empty.yaml'
    var_12 = ''
    var_13 = 'config_env.yaml'
    var_14 = 'cookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: $HOME/.replay\n'
    var_15 = '$HOME/.cookiecutters'
    var_16 = '$HOME/.replay'
    var_17 = 'partial.yaml'
    var_18 = 'default_context:\n  project_name: my_project\n'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = 'non_existent.yaml'
    var_4 = 'invalid.yaml'
    var_5 = '{ invalid yaml content: ['
    var_6 = 'non_dict.yaml'
    var_7 = '- item1\n- item2'
    var_8 = 'empty.yaml'
    var_9 = ''
    var_10 = 'config_env.yaml'
    var_11 = 'cookiecutters_dir: $HOME/.cookiecutters'
    var_12 = '~/.cookiecutters'
    var_13 = 'config_tilde.yaml'
    var_14 = 'replay_dir: ~/my_replay'
    var_15 = '~/my_replay'
    var_16 = 'partial.yaml'
    var_17 = 'default_context:\n  key: value'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir: /custom/path\n'
    var_5 = 'empty.yaml'
    var_6 = ''
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: ['
    var_9 = 'non_dict.yaml'
    var_10 = '- item1\n- item2\n'
    var_11 = 'config_env.yaml'
    var_12 = 'cookiecutters_dir: $HOME/.custom\n'
    var_13 = 'HOME'
    var_14 = '/home/user'
    var_15 = 'config_tilde.yaml'
    var_16 = 'replay_dir: ~/custom_replay\n'
    var_17 = 'custom_abbrev.yaml'
    var_18 = 'abbreviations:\n  custom: https://custom.com/{0}\n'
    var_19 = 'context.yaml'
    var_20 = 'default_context:\n  author: Test Author\n'



# Parsed testcases at query #17
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/to/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'test_config.yaml'
    var_4 = 'cookiecutters_dir'
    var_5 = 'replay_dir'
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'test_config.yaml'
    var_10 = 'cookiecutters_dir'
    var_11 = 'replay_dir'
    var_12 = '$TEST_DIR/cookies'
    var_13 = '$TEST_DIR/replay'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'test_config.yaml'
    var_16 = 'abbreviations'
    var_17 = 'custom'
    var_18 = 'https://custom.com/{0}.git'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'invalid_config.yaml'
    var_22 = '{ invalid yaml: ['
    var_23 = 'list_config.yaml'
    var_24 = 'item1'
    var_25 = 'item2'
    var_26 = [var_24, var_25]
    var_27 = 'empty_config.yaml'
    var_28 = ''
    var_29 = 'test_config.yaml'
    var_30 = var_28 / var_29
    var_31 = 'default_context'
    var_32 = 'project_name'
    var_33 = 'test'
    var_34 = {var_32: var_33}
    var_35 = {var_31: var_34}
    var_36 = module_0.get_config(var_30)
    var_37 = 'partial_config.yaml'
    var_38 = 'default_context'
    var_39 = 'author'
    var_40 = 'Test Author'
    var_41 = {var_39: var_40}
    var_42 = {var_38: var_41}
    var_43 = module_0.get_config(var_30)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = '~/.cookiecutters/'
    var_4 = '~/.cookiecutter_replay/'
    var_5 = 'cookiecutters_dir'
    var_6 = '/custom/path'
    var_7 = {var_5: var_6}
    var_8 = module_0.get_user_config(default_config=var_7)
    var_9 = 'custom_config.yaml'
    var_10 = 'cookiecutters_dir: /tmp/custom\nreplay_dir: /tmp/replay'
    var_11 = 'env_config.yaml'
    var_12 = 'cookiecutters_dir: /env/path'
    var_13 = 'COOKIECUTTER_CONFIG'
    var_14 = False
    var_15 = module_0.get_user_config()
    var_16 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_17 = '/nonexistent/path'
    var_18 = module_0.get_user_config()
    var_19 = '.cookiecutterrc'
    var_20 = 'default_context:\n  author: Test Author'
    var_21 = module_0.get_user_config()
    var_22 = '/nonexistent/env/config.yaml'
    var_23 = module_0.get_user_config()
    var_24 = '/nonexistent/custom/config.yaml'
    var_25 = module_0.get_user_config(var_24)
    var_26 = 'config.yaml'
    var_27 = 'cookiecutters_dir: /file/path'
    var_28 = '/dict/path'
    var_29 = {var_5: var_28}
    var_30 = module_0.get_user_config(default_config=var_25)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yml'
    var_8 = 'cookiecutters_dir: /my/custom/path\n'
    var_9 = 'env_config.yml'
    var_10 = 'replay_dir: /env/replay/path\n'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'os.path.exists'
    var_15 = lambda x: var_12
    var_16 = module_0.get_user_config()
    var_17 = '.cookiecutterrc'
    var_18 = 'abbreviations:\n  custom: https://example.com/{0}\n'
    var_19 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_20 = module_0.get_user_config()
    var_21 = '/nonexistent/path/config.yml'
    var_22 = module_0.get_user_config(var_21)
    var_23 = '/nonexistent/env/config.yml'
    var_24 = module_0.get_user_config()
    var_25 = '/priority/path'
    var_26 = {var_3: var_25}
    var_27 = '/some/file.yml'
    var_28 = module_0.get_user_config(var_27, var_26)
    var_29 = module_0.get_user_config(var_27, var_22)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid YAML configuration file.'
    var_1 = '\ncookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\ndefault_context:\n  project_name: My Project\nabbreviations:\n  gh: https://github.com/{0}.git\n'

import cookiecutter.config as module_0

def test_case_0():
    var_0 = "Test get_config raises ConfigDoesNotExistException when file doesn't exist."
    var_1 = '/nonexistent/path/to/config.yml'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration for invalid YAML.'
    var_1 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises InvalidConfiguration when YAML top-level is not a dict.'
    var_1 = '- item1\n- item2\n'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns default config.'
    var_1 = ''
    var_2 = '~/.cookiecutters/'
    var_3 = '~/.cookiecutter_replay/'

def test_case_0():
    var_0 = 'Test get_config expands environment variables and user home.'
    var_1 = '\ncookiecutters_dir: $HOME/.my_cookiecutters/\nreplay_dir: ~/my_replay/\n'
    var_2 = '~/.my_cookiecutters/'
    var_3 = '~/my_replay/'

def test_case_0():
    var_0 = 'Test get_config merges provided config with default values.'
    var_1 = '\ncookiecutters_dir: ~/.custom_cookiecutters/\n'
    var_2 = '~/.custom_cookiecutters/'

def test_case_0():
    var_0 = 'Test get_config merges custom abbreviations with defaults.'
    var_1 = '\nabbreviations:\n  custom: https://custom.example.com/{0}\n'



# Parsed testcases at query #21
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various configurations.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yaml'
    var_8 = 'cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay'
    var_9 = 'COOKIECUTTER_CONFIG'
    var_10 = False
    var_11 = 'HOME'
    var_12 = 'env_config.yaml'
    var_13 = 'cookiecutters_dir: /env/cookiecutters'
    var_14 = module_0.get_user_config()
    var_15 = 'fake_home'
    var_16 = module_0.get_user_config()
    var_17 = '.cookiecutterrc'
    var_18 = 'replay_dir: /user/replay'
    var_19 = module_0.get_user_config()
    var_20 = 'abbreviations'
    var_21 = 'custom'
    var_22 = 'https://custom.com/{0}'
    var_23 = {var_21: var_22}
    var_24 = {var_20: var_23}
    var_25 = module_0.get_user_config(default_config=var_24)
    var_26 = 'path_expand_config.yaml'
    var_27 = 'cookiecutters_dir: ~/expanded_path'
    var_28 = 'invalid_config.yaml'
    var_29 = 'invalid: yaml: content: ['
    var_30 = module_0.get_user_config(var_0)
    var_31 = '/non/existent/path.yaml'
    var_32 = module_0.get_user_config(var_31)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yaml'
    var_8 = 'cookiecutters_dir: /tmp/custom\nreplay_dir: /tmp/replay\n'
    var_9 = 'env_config.yaml'
    var_10 = 'cookiecutters_dir: /env/path\n'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'os.path.exists'
    var_15 = lambda x: var_12
    var_16 = module_0.get_user_config()
    var_17 = 'user_config.yaml'
    var_18 = 'cookiecutters_dir: /user/path\n'
    var_19 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_20 = module_0.get_user_config()
    var_21 = 'non_existent.yaml'
    var_22 = module_0.get_user_config(var_0)
    var_23 = 'invalid.yaml'
    var_24 = 'invalid: yaml: content:'
    var_25 = module_0.get_user_config(var_0)
    var_26 = 'non_dict.yaml'
    var_27 = '- item1\n- item2\n'
    var_28 = module_0.get_user_config(var_0)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid YAML configuration file.'
    var_1 = 'cookiecutterrc'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = 'default_context'
    var_5 = 'abbreviations'
    var_6 = '~/.cookiecutters/'
    var_7 = '~/.cookiecutter_replay/'
    var_8 = {}
    var_9 = 'gh'
    var_10 = 'https://github.com/{0}.git'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_11}

import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config raises exception when config file does not exist.'
    var_1 = '/nonexistent/path/to/config'
    var_2 = module_0.get_config(var_1)

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'cookiecutterrc'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when top-level YAML is not a dict.'
    var_1 = 'cookiecutterrc'
    var_2 = 'list'
    var_3 = 'not'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'Test get_config expands environment variables in paths.'
    var_1 = 'cookiecutterrc'
    var_2 = 'cookiecutters_dir'
    var_3 = 'replay_dir'
    var_4 = '$HOME/.cookiecutters/'
    var_5 = '${HOME}/.cookiecutter_replay/'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'Test get_config merges loaded config with default config.'
    var_1 = 'cookiecutterrc'
    var_2 = 'cookiecutters_dir'
    var_3 = '/custom/path/'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns defaults.'
    var_1 = 'cookiecutterrc'
    var_2 = ''
    var_3 = '~/.cookiecutters/'
    var_4 = '~/.cookiecutter_replay/'

def test_case_0():
    var_0 = 'Test get_config preserves and merges nested abbreviations.'
    var_1 = 'cookiecutterrc'
    var_2 = 'abbreviations'
    var_3 = 'custom'
    var_4 = 'https://custom.com/{0}.git'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #24
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/to/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = 'abbreviations'
    var_7 = '~/.custom_cookiecutters/'
    var_8 = '~/.custom_replay/'
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'custom'
    var_13 = 'https://example.com/{0}'
    var_14 = {var_12: var_13}
    var_15 = {var_3: var_7, var_4: var_8, var_5: var_11, var_6: var_14}
    var_16 = '~/.custom_cookiecutters/'
    var_17 = '~/.custom_replay/'
    var_18 = 'cookiecutters_dir'
    var_19 = 'replay_dir'
    var_20 = '$HOME/.cookiecutters/'
    var_21 = '$HOME/.replay/'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = ''
    var_24 = 'invalid: yaml: content: ['
    var_25 = 'list'
    var_26 = 'not'
    var_27 = 'dict'
    var_28 = [var_25, var_26, var_27]
    var_29 = 'cookiecutters_dir'
    var_30 = '~/.custom/'
    var_31 = {var_29: var_30}
    var_32 = 'cookiecutters_dir'
    var_33 = '~/.my_cookiecutters/'
    var_34 = {var_32: var_33}
    var_35 = '~/.my_cookiecutters/'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'valid_config.yaml'
    var_2 = '\ncookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\ndefault_context:\n  full_name: John Doe\nabbreviations:\n  gh: https://github.com/{0}.git\n'
    var_3 = 'env_config.yaml'
    var_4 = '\ncookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: ~/replay\n'
    var_5 = 'empty_config.yaml'
    var_6 = ''
    var_7 = 'nonexistent.yaml'
    var_8 = 'invalid.yaml'
    var_9 = '{ invalid yaml: ['
    var_10 = 'non_dict.yaml'
    var_11 = '- item1\n- item2'
    var_12 = 'partial_config.yaml'
    var_13 = '\ndefault_context:\n  author_name: Jane Smith\n'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.config as module_0
import yaml as module_1

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = '/nonexistent/path/config.yaml'
    var_2 = module_0.get_config(var_1)
    var_3 = 'config.yaml'
    var_4 = 'cookiecutters_dir'
    var_5 = 'replay_dir'
    var_6 = 'default_context'
    var_7 = 'abbreviations'
    var_8 = '~/.cookiecutters/'
    var_9 = '~/.cookiecutter_replay/'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = 'custom'
    var_14 = 'https://example.com/{0}.git'
    var_15 = {var_13: var_14}
    var_16 = {var_4: var_8, var_5: var_9, var_6: var_12, var_7: var_15}
    var_17 = module_1.dump(var_16)
    var_18 = 'utf-8'
    var_19 = 'invalid.yaml'
    var_20 = 'invalid: yaml: content:'
    var_21 = 'non_dict.yaml'
    var_22 = '- item1\n- item2'
    var_23 = 'empty.yaml'
    var_24 = ''
    var_25 = 'config_vars.yaml'
    var_26 = 'cookiecutters_dir: $HOME/.my_cookiecutters/\nreplay_dir: ~/replay/'
    var_27 = 'partial.yaml'
    var_28 = 'default_context:\n  key: value'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\ndefault_context:\n  author_name: John Doe\nabbreviations:\n  custom: https://custom.com/{0}\n'
    var_3 = 'config2.yaml'
    var_4 = '\ncookiecutters_dir: $HOME/.cookiecutters/\nreplay_dir: ~/custom_replay/\n'
    var_5 = '$HOME'
    var_6 = 'non_existent.yaml'
    var_7 = 'invalid.yaml'
    var_8 = 'invalid: yaml: content: ['
    var_9 = 'non_dict.yaml'
    var_10 = '- item1\n- item2'
    var_11 = 'empty.yaml'
    var_12 = ''
    var_13 = 'partial.yaml'
    var_14 = 'cookiecutters_dir: /custom/path/'



# Parsed testcases at query #28
#--------------------------


import yaml as module_0

def test_case_0():
    var_0 = 'Test get_config function with various scenarios.'
    var_1 = 'non_existent.yaml'
    var_2 = 'config.yaml'
    var_3 = 'cookiecutters_dir'
    var_4 = 'replay_dir'
    var_5 = 'default_context'
    var_6 = 'abbreviations'
    var_7 = '/custom/cookiecutters'
    var_8 = '/custom/replay'
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'custom'
    var_13 = 'https://example.com/{0}.git'
    var_14 = {var_12: var_13}
    var_15 = {var_3: var_7, var_4: var_8, var_5: var_11, var_6: var_14}
    var_16 = module_0.dump(var_15)
    var_17 = 'config_vars.yaml'
    var_18 = '~/.cookiecutters'
    var_19 = '$HOME/.cookiecutter_replay'
    var_20 = {var_3: var_18, var_4: var_19}
    var_21 = module_0.dump(var_20)
    var_22 = 'invalid.yaml'
    var_23 = '{ invalid yaml content: ['
    var_24 = 'non_dict.yaml'
    var_25 = '- item1\n- item2'
    var_26 = 'empty.yaml'
    var_27 = ''
    var_28 = 'partial.yaml'
    var_29 = '/partial/path'
    var_30 = {var_3: var_29}
    var_31 = module_0.dump(var_30)
    var_32 = '~/.cookiecutter_replay/'
    var_33 = 'string_path.yaml'
    var_34 = '/test'
    var_35 = {var_3: var_34}
    var_36 = module_0.dump(var_35)



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test get_config function with valid and invalid configurations.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay\ndefault_context:\n    project_name: my_project\nabbreviations:\n    custom: https://example.com/{0}.git\n'

def test_case_0():
    var_0 = 'Test get_config raises exception for non-existent file.'
    var_1 = 'nonexistent.yaml'

def test_case_0():
    var_0 = 'Test get_config raises exception for invalid YAML.'
    var_1 = 'invalid.yaml'
    var_2 = 'invalid: yaml: content: ['

def test_case_0():
    var_0 = 'Test get_config raises exception when YAML is not a dict.'
    var_1 = 'list.yaml'
    var_2 = '- item1\n- item2'

def test_case_0():
    var_0 = 'Test get_config with empty YAML file returns defaults.'
    var_1 = 'empty.yaml'
    var_2 = ''
    var_3 = '~/.cookiecutters/'
    var_4 = '~/.cookiecutter_replay/'

def test_case_0():
    var_0 = 'Test get_config expands environment variables and user home.'
    var_1 = 'config.yaml'
    var_2 = '\ncookiecutters_dir: ~/my_cookiecutters\nreplay_dir: $HOME/.replay\n'
    var_3 = '~/my_cookiecutters'

def test_case_0():
    var_0 = 'Test get_config merges partial config with defaults.'
    var_1 = 'partial.yaml'
    var_2 = '\ndefault_context:\n    author: John Doe\n'
    var_3 = '~/.cookiecutters/'



# Parsed testcases at query #30
#--------------------------


import cookiecutter.config as module_0

def test_case_0():
    var_0 = 'Test get_user_config function with various scenarios.'
    var_1 = True
    var_2 = module_0.get_user_config(default_config=var_1)
    var_3 = 'cookiecutters_dir'
    var_4 = '/custom/path'
    var_5 = {var_3: var_4}
    var_6 = module_0.get_user_config(default_config=var_5)
    var_7 = 'custom_config.yml'
    var_8 = 'cookiecutters_dir: /custom/cookiecutters\n'
    var_9 = 'env_config.yml'
    var_10 = 'replay_dir: /env/replay\n'
    var_11 = 'COOKIECUTTER_CONFIG'
    var_12 = False
    var_13 = module_0.get_user_config()
    var_14 = 'HOME'
    var_15 = module_0.get_user_config()
    var_16 = '.cookiecutterrc'
    var_17 = "abbreviations:\n  custom: 'https://custom.com/{0}'\n"
    var_18 = 'cookiecutter.config.USER_CONFIG_PATH'
    var_19 = module_0.get_user_config()
    var_20 = 'custom_key'
    var_21 = 'custom_value'
    var_22 = {var_20: var_21}
    var_23 = 'invalid.yml'
    var_24 = 'invalid: yaml: content: :\n'
    var_25 = module_0.get_user_config(var_18)
    var_26 = '/non/existent/path.yml'
    var_27 = module_0.get_user_config(var_26)
    var_28 = 'my_config.yml'
    var_29 = 'cookiecutters_dir: /my/path\n'



