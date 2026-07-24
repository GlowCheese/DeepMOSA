####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 'multi_line_output'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = '88'
    var_5 = 'pyproject.toml'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '3'
    var_8 = 'cli'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = "isort was provided settings that it doesn't support:\n\n"
    var_13 = "\t- line_length = 88  (source: 'pyproject.toml')"
    var_14 = "\t- multi_line_output = 3  (source: 'cli')"
    var_15 = '\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n'
    var_16 = var_11.unsupported_settings
    var_17 = bool(var_11.unsupported_settings == var_10)
    assert var_17 is True
    var_18 = str(var_11)
    var_19 = var_12 + var_13 + '\n' + var_14 + var_15
    var_20 = bool(var_12 + var_13 + '\n' + var_14 + var_15 in var_18)
    assert var_20 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = var_1.unsupported_settings
    var_3 = bool(var_1.unsupported_settings == {})
    assert var_3 is True
    var_4 = str(var_1)
    var_5 = "isort was provided settings that it doesn't support:\n\n\n\nFor a complete"
    var_6 = bool("isort was provided settings that it doesn't support:\n\n\n\nFor a complete" in var_4)
    assert var_6 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'line_length'
    var_1 = 'multi_line_output'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = '88'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '3'
    var_8 = 'cli'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = "isort was provided settings that it doesn't support:\n\n\t- line_length = 88  (source: 'config')\n\t- multi_line_output = 3  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_13 = str(var_11)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True
    var_15 = var_11.unsupported_settings
    var_16 = bool(var_11.unsupported_settings == var_10)
    assert var_16 is True

import isort.exceptions as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.UnsupportedSettings(var_0)
    var_2 = "isort was provided settings that it doesn't support:\n\n\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_3 = str(var_1)
    var_4 = bool(var_3 == var_2)
    assert var_4 is True
    var_5 = var_1.unsupported_settings
    var_6 = bool(var_1.unsupported_settings == {})
    assert var_6 is True



