####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'profile'
    var_1 = 'line_length'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'black'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 88
    var_8 = 'cli'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = "isort was provided settings that it doesn't support:\n\n\t- profile = black  (source: 'config')\n\t- line_length = 88  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_13 = var_11.unsupported_settings
    var_14 = bool(var_11.unsupported_settings == var_10)
    assert var_14 is True
    var_15 = str(var_11)
    var_16 = bool(var_15 == var_12)
    assert var_16 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = "isort was provided settings that it doesn't support:\n\n\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_3 = var_1.unsupported_settings
    var_4 = bool(var_1.unsupported_settings == {})
    assert var_4 is True
    var_5 = str(var_1)
    var_6 = bool(var_5 == var_2)
    assert var_6 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'some_option'
    var_1 = 'another_option'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'some_value'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True
    var_8 = 'cli'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = "\t- some_option = some_value  (source: 'config')"
    var_13 = "\t- another_option = True  (source: 'cli')"
    var_14 = var_11.unsupported_settings
    var_15 = bool(var_11.unsupported_settings == var_10)
    assert var_15 is True
    var_16 = str(var_11)
    var_17 = bool(var_12 in var_16)
    assert var_17 is True
    var_18 = str(var_11)
    var_19 = bool(var_13 in var_18)
    assert var_19 is True
    var_20 = str(var_11)
    var_21 = "isort was provided settings that it doesn't support:"
    var_22 = bool("isort was provided settings that it doesn't support:" in var_20)
    assert var_22 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = var_1.unsupported_settings
    var_3 = bool(var_1.unsupported_settings == {})
    assert var_3 is True
    var_4 = str(var_1)
    var_5 = "isort was provided settings that it doesn't support:\n\n\n"
    var_6 = bool("isort was provided settings that it doesn't support:\n\n\n" in var_4)
    assert var_6 is True



