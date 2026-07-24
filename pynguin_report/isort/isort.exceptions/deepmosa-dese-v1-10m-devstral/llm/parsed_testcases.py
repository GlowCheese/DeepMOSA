####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_5 = 'source1'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'value2'
    var_8 = 'source2'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = str(var_11)
    assert var_12 == "isort was provided settings that it doesn't support:\n\n\t- setting1 = value1  (source: 'source1')\n\t- setting2 = value2  (source: 'source2')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.exceptions as module_0

def test_case_0():
    var_0 = 'setting1'
    var_1 = 'setting2'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'invalid_value'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 42
    var_8 = 'CLI'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = module_0.UnsupportedSettings(var_10)
    var_12 = str(var_11)
    assert var_12 == "isort was provided settings that it doesn't support:\n\n\t- setting1 = invalid_value  (source: 'config')\n\t- setting2 = 42  (source: 'CLI')\n\nFor a complete and up-to-date listing of supported settings see: https://pycqa.github.io/isort/docs/configuration/options.\n"



