####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unsupported_settings_constructor_single_setting. Retrieved 9/10 statements.


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'unknown_option'
    var_1 = 'invalid_setting'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'test_value'
    var_5 = 'config.ini'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 42
    var_8 = 'CLI'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = var_11.unsupported_settings
    var_13 = bool(var_11.unsupported_settings == var_10)
    assert var_13 is True
    var_14 = str(var_11)
    var_15 = "isort was provided settings that it doesn't support:"
    var_16 = bool("isort was provided settings that it doesn't support:" in var_14)
    assert var_16 is True
    var_17 = str(var_11)
    var_18 = "unknown_option = test_value  (source: 'config.ini')"
    var_19 = bool("unknown_option = test_value  (source: 'config.ini')" in var_17)
    assert var_19 is True
    var_20 = str(var_11)
    var_21 = "invalid_setting = 42  (source: 'CLI')"
    var_22 = bool("invalid_setting = 42  (source: 'CLI')" in var_20)
    assert var_22 is True
    var_23 = str(var_11)
    var_24 = 'https://pycqa.github.io/isort/docs/configuration/options'
    var_25 = bool('https://pycqa.github.io/isort/docs/configuration/options' in var_23)
    assert var_25 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = var_1.unsupported_settings
    var_3 = bool(var_1.unsupported_settings == var_0)
    assert var_3 is True
    var_4 = str(var_1)
    var_5 = "isort was provided settings that it doesn't support:"
    var_6 = bool("isort was provided settings that it doesn't support:" in var_4)
    assert var_6 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'bad_option'
    var_1 = 'value'
    var_2 = 'source'
    var_3 = 'some_value'
    var_4 = 'environment'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.UnsupportedSettings(var_6)
    var_8 = var_7.unsupported_settings
    var_9 = bool(var_7.unsupported_settings == var_6)
    assert var_9 is True
    var_10 = str(var_7)
    var_11 = "bad_option = some_value  (source: 'environment')"
    var_12 = bool("bad_option = some_value  (source: 'environment')" in var_10)
    assert var_12 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting1'
    var_1 = 'setting2'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'val1'
    var_5 = 'config.ini'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'val2'
    var_8 = 'command_line'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = var_11.unsupported_settings
    var_13 = bool(var_11.unsupported_settings == var_10)
    assert var_13 is True
    var_14 = str(var_11)
    var_15 = "isort was provided settings that it doesn't support:"
    var_16 = bool("isort was provided settings that it doesn't support:" in var_14)
    assert var_16 is True
    var_17 = str(var_11)
    var_18 = "setting1 = val1  (source: 'config.ini')"
    var_19 = bool("setting1 = val1  (source: 'config.ini')" in var_17)
    assert var_19 is True
    var_20 = str(var_11)
    var_21 = "setting2 = val2  (source: 'command_line')"
    var_22 = bool("setting2 = val2  (source: 'command_line')" in var_20)
    assert var_22 is True
    var_23 = str(var_11)
    var_24 = 'https://pycqa.github.io/isort/docs/configuration/options'
    var_25 = bool('https://pycqa.github.io/isort/docs/configuration/options' in var_23)
    assert var_25 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = var_1.unsupported_settings
    var_3 = bool(var_1.unsupported_settings == {})
    assert var_3 is True
    var_4 = str(var_1)
    var_5 = "isort was provided settings that it doesn't support:"
    var_6 = bool("isort was provided settings that it doesn't support:" in var_4)
    assert var_6 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'unknown_option'
    var_1 = 'value'
    var_2 = 'source'
    var_3 = 'some_value'
    var_4 = 'pyproject.toml'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.UnsupportedSettings(var_6)
    var_8 = var_7.unsupported_settings
    var_9 = bool(var_7.unsupported_settings == var_6)
    assert var_9 is True
    var_10 = str(var_7)
    var_11 = "unknown_option = some_value  (source: 'pyproject.toml')"
    var_12 = bool("unknown_option = some_value  (source: 'pyproject.toml')" in var_10)
    assert var_12 is True



