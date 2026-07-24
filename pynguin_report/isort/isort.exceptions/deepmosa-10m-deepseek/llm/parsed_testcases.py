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
    var_12 = "isort was provided settings that it doesn't support:\n\n\t- setting1 = value1  (source: 'config')\n\t- setting2 = value2  (source: 'cli')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"
    var_13 = str(var_11)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True
    var_15 = var_11.unsupported_settings
    var_16 = bool(var_11.unsupported_settings == var_10)
    assert var_16 is True



