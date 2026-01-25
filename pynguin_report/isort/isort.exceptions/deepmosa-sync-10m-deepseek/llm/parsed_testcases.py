####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting1'
    var_1 = 'setting2'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'value1'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'value2'
    var_8 = 'cli'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = var_11.unsupported_settings
    var_13 = bool(var_11.unsupported_settings == var_10)
    assert var_13 is True
    var_14 = "isort was provided settings that it doesn't support:\n\n\t- setting1 = value1  (source: 'config')\n\t- setting2 = value2  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_15 = str(var_11)
    var_16 = bool(var_15 == var_14)
    assert var_16 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_unsupported_settings_constructor_check_exception_inheritance. Retrieved 8/9 statements.


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting1'
    var_1 = 'value'
    var_2 = 'source'
    var_3 = 'value1'
    var_4 = 'config'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.UnsupportedSettings(var_6)
    var_8 = var_7.unsupported_settings
    var_9 = bool(var_7.unsupported_settings == var_6)
    assert var_9 is True
    var_10 = str(var_7)
    var_11 = "isort was provided settings that it doesn't support:"
    var_12 = bool("isort was provided settings that it doesn't support:" in var_10)
    assert var_12 is True
    var_13 = str(var_7)
    var_14 = "setting1 = value1  (source: 'config')"
    var_15 = bool("setting1 = value1  (source: 'config')" in var_13)
    assert var_15 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting1'
    var_1 = 'setting2'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'value1'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'value2'
    var_8 = 'cli'
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
    var_17 = "setting1 = value1  (source: 'config')"
    var_18 = bool("setting1 = value1  (source: 'config')" in var_14)
    assert var_18 is True
    var_19 = "setting2 = value2  (source: 'cli')"
    var_20 = bool("setting2 = value2  (source: 'cli')" in var_14)
    assert var_20 is True

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
    var_0 = 'setting1'
    var_1 = 'value'
    var_2 = 'source'
    var_3 = 'value1'
    var_4 = 'config'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.UnsupportedSettings(var_6)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting@name'
    var_1 = 'value'
    var_2 = 'source'
    var_3 = 'value@123'
    var_4 = 'file.json'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.UnsupportedSettings(var_6)
    var_8 = var_7.unsupported_settings
    var_9 = bool(var_7.unsupported_settings == var_6)
    assert var_9 is True
    var_10 = str(var_7)
    var_11 = "setting@name = value@123  (source: 'file.json')"
    var_12 = bool("setting@name = value@123  (source: 'file.json')" in var_10)
    assert var_12 is True



