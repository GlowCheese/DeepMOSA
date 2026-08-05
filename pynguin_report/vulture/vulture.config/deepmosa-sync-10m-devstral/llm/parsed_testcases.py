####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'unknown_key'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_invalid_toml_type. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['verbose']
    assert var_5 is True
    var_6 = var_4['paths']
    var_7 = bool(var_4['paths'] == ['path1', 'path2'])
    assert var_7 is True
    var_8 = var_4['exclude']
    var_9 = bool(var_4['exclude'] == [])
    assert var_9 is True
    var_10 = var_4['ignore_decorators']
    var_11 = bool(var_4['ignore_decorators'] == [])
    assert var_11 is True
    var_12 = var_4['ignore_names']
    var_13 = bool(var_4['ignore_names'] == [])
    assert var_13 is True
    var_14 = var_4['make_whitelist']
    assert var_14 is False
    var_15 = var_4['min_confidence']
    assert var_15 == 60
    var_16 = var_4['sort_by_size']
    assert var_16 is False

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '
    var_1 = '--verbose'
    var_2 = 'path3'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['verbose']
    assert var_2 is False
    var_3 = var_1['paths']
    var_4 = bool(var_1['paths'] == [])
    assert var_4 is True
    var_5 = var_1['exclude']
    var_6 = bool(var_1['exclude'] == [])
    assert var_6 is True
    var_7 = var_1['ignore_decorators']
    var_8 = bool(var_1['ignore_decorators'] == [])
    assert var_8 is True
    var_9 = var_1['ignore_names']
    var_10 = bool(var_1['ignore_names'] == [])
    assert var_10 is True
    var_11 = var_1['make_whitelist']
    assert var_11 is False
    var_12 = var_1['min_confidence']
    assert var_12 == 60
    var_13 = var_1['sort_by_size']
    assert var_13 is False

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_1 = bool(False)
    assert var_1 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_invalid_type_in_toml_raises_error. Retrieved 1/5 statements.
# Partially parsed test_make_config_unknown_key_in_toml_raises_error. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['path1'])
    assert var_4 is True
    var_5 = var_2['exclude']
    var_6 = bool(var_2['exclude'] == [])
    assert var_6 is True
    var_7 = var_2['ignore_decorators']
    var_8 = bool(var_2['ignore_decorators'] == [])
    assert var_8 is True
    var_9 = var_2['ignore_names']
    var_10 = bool(var_2['ignore_names'] == [])
    assert var_10 is True
    var_11 = var_2['make_whitelist']
    assert var_11 is False
    var_12 = var_2['min_confidence']
    assert var_12 == 60
    var_13 = var_2['sort_by_size']
    assert var_13 is False
    var_14 = var_2['verbose']
    assert var_14 is False
    var_15 = var_2['config']
    assert var_15 == 'pyproject.toml'

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--exclude'
    var_3 = '*.py'
    var_4 = '--ignore-decorators'
    var_5 = 'decorator1'
    var_6 = '--ignore-names'
    var_7 = 'name1'
    var_8 = '--make-whitelist'
    var_9 = '--min-confidence'
    var_10 = '80'
    var_11 = '--sort-by-size'
    var_12 = '--verbose'
    var_13 = '--config'
    var_14 = 'custom.toml'
    var_15 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = var_16['paths']
    var_18 = bool(var_16['paths'] == ['path1', 'path2'])
    assert var_18 is True
    var_19 = var_16['exclude']
    var_20 = bool(var_16['exclude'] == ['*.py'])
    assert var_20 is True
    var_21 = var_16['ignore_decorators']
    var_22 = bool(var_16['ignore_decorators'] == ['decorator1'])
    assert var_22 is True
    var_23 = var_16['ignore_names']
    var_24 = bool(var_16['ignore_names'] == ['name1'])
    assert var_24 is True
    var_25 = var_16['make_whitelist']
    assert var_25 is True
    var_26 = var_16['min_confidence']
    assert var_26 == 80
    var_27 = var_16['sort_by_size']
    assert var_27 is True
    var_28 = var_16['verbose']
    assert var_28 is True
    var_29 = var_16['config']
    assert var_29 == 'custom.toml'

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["*.py"]\n    ignore_decorators = ["decorator1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 80\n    sort_by_size = true\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    exclude = ["*.py"]\n    '
    var_1 = 'cli_path'
    var_2 = '--exclude'
    var_3 = '*.txt'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = "not_a_list"\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '



# Parsed testcases at query #4
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 1
    var_4 = 'value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = ''
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = 'test.toml'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['verbose']
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'unknown_key'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_4)
    var_8 = bool(False)
    assert var_8 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_verbose_message_when_toml_detected. Retrieved 3/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = 'Reading configuration from test.toml'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #11
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 1
    var_4 = 'value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = ''
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)
    assert var_11 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = var_0['paths']
    var_2 = bool(var_0['paths'] == [])
    assert var_2 is True
    var_3 = var_0['exclude']
    var_4 = bool(var_0['exclude'] == [])
    assert var_4 is True
    var_5 = var_0['ignore_decorators']
    var_6 = bool(var_0['ignore_decorators'] == [])
    assert var_6 is True
    var_7 = var_0['ignore_names']
    var_8 = bool(var_0['ignore_names'] == [])
    assert var_8 is True
    var_9 = var_0['make_whitelist']
    assert var_9 is False
    var_10 = var_0['min_confidence']
    assert var_10 == 60
    var_11 = var_0['sort_by_size']
    assert var_11 is False
    var_12 = var_0['verbose']
    assert var_12 is False

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--verbose'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['exclude']
    var_8 = bool(var_6['exclude'] == ['test_*.py'])
    assert var_8 is True
    var_9 = var_6['min_confidence']
    assert var_9 == 80
    var_10 = var_6['verbose']
    assert var_10 is True

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    '
    var_1 = '--exclude'
    var_2 = 'docs'
    var_3 = '--min-confidence'
    var_4 = '90'
    var_5 = [var_1, var_2, var_3, var_4]

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()



# Parsed testcases at query #14
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['verbose']
    assert var_4 is True
    var_5 = var_3['config']
    assert var_5 == 'test.toml'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 3/6 statements.
# Partially parsed test_make_config_unknown_key_in_toml_raises_error. Retrieved 1/6 statements.
# Partially parsed test_make_config_wrong_type_in_toml_raises_error. Retrieved 1/6 statements.
# Partially parsed test_make_config_verbose_shows_toml_path. Retrieved 1/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['path1', 'path2'])
    assert var_5 is True
    var_6 = var_3['exclude']
    var_7 = bool(var_3['exclude'] == [])
    assert var_7 is True
    var_8 = var_3['ignore_decorators']
    var_9 = bool(var_3['ignore_decorators'] == [])
    assert var_9 is True
    var_10 = var_3['ignore_names']
    var_11 = bool(var_3['ignore_names'] == [])
    assert var_11 is True
    var_12 = var_3['make_whitelist']
    assert var_12 is False
    var_13 = var_3['min_confidence']
    assert var_13 == 60
    var_14 = var_3['sort_by_size']
    assert var_14 is False
    var_15 = var_3['verbose']
    assert var_15 is False

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["*test*.py"]\n    ignore_decorators = ["@decorator1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 80\n    sort_by_size = true\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    min_confidence = 80\n    '
    var_1 = 'cli_path'
    var_2 = [var_1]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = "not_a_list"\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    verbose = true\n    '



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #17
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_config_toml_overrides_defaults. Retrieved 1/5 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/8 statements.
# Partially parsed test_make_config_verbose_shows_toml_path. Retrieved 3/7 statements.
# Partially parsed test_make_config_toml_invalid_key_raises_error. Retrieved 1/6 statements.
# Partially parsed test_make_config_toml_wrong_type_raises_error. Retrieved 1/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['min_confidence']
    assert var_4 == 50
    var_5 = var_3['paths']
    var_6 = var_3['verbose']

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 75\npaths = ["src/"]\n'

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 75\npaths = ["src/"]\n'
    var_1 = '--min-confidence'
    var_2 = '90'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 75\n'
    var_1 = '--verbose'
    var_2 = [var_1]

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_integer'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = '[tool.vulture]\ninvalid_key = "value"\n'

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = "not_an_integer"\n'



# Parsed testcases at query #20
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 42
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = False
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = var_4[var_0]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_7[var_0]
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_0.type(*var_13, **var_14)
    var_16 = bool(var_11 is var_15)
    assert var_16 is True
    var_17 = var_4[var_1]
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.type(*var_18, **var_19)
    var_21 = var_7[var_1]
    var_22 = [var_21]
    var_23 = {}
    var_24 = module_0.type(*var_22, **var_23)
    var_25 = bool(var_20 is var_24)
    assert var_25 is True



# Parsed testcases at query #21
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['verbose']
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = '[tool.vulture]\n'



# Parsed testcases at query #24
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 1
    var_4 = 'value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = ''
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'valid_toml_file.toml'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 6/9 statements.
# Partially parsed test_make_config_with_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_wrong_type_in_toml. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--verbose'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['exclude']
    var_8 = bool(var_6['exclude'] == ['test_*.py'])
    assert var_8 is True
    var_9 = var_6['verbose']
    assert var_9 is True
    var_10 = var_6['paths']
    var_11 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_11 is True
    var_12 = var_6['min_confidence']
    assert var_12 == 60
    var_13 = var_6['sort_by_size']
    assert var_13 is False

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\npaths = ["path1"]\nmin_confidence = 10\nsort_by_size = true\n'
    var_1 = '--exclude'
    var_2 = 'test_*.py'
    var_3 = '--verbose'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\npaths = ["path1"]\nmin_confidence = 10\nsort_by_size = true\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_1 = bool(False)
    assert var_1 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_1 = bool(False)
    assert var_1 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['exclude']
    var_4 = bool(var_2['exclude'] == [])
    assert var_4 is True
    var_5 = var_2['ignore_decorators']
    var_6 = bool(var_2['ignore_decorators'] == [])
    assert var_6 is True
    var_7 = var_2['ignore_names']
    var_8 = bool(var_2['ignore_names'] == [])
    assert var_8 is True
    var_9 = var_2['make_whitelist']
    assert var_9 is False
    var_10 = var_2['min_confidence']
    assert var_10 == 60
    var_11 = var_2['sort_by_size']
    assert var_11 is False
    var_12 = var_2['verbose']
    assert var_12 is False
    var_13 = var_2['paths']
    var_14 = bool(var_2['paths'] == ['path1'])
    assert var_14 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = bool(var_1 == {'paths': [], 'config': 'pyproject.toml'})
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': ['file1.py', 'file2.py'], 'config': 'pyproject.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*,venv'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'exclude': ['test_*', 'venv'], 'config': 'pyproject.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@app.route,@require_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_decorators': ['@app.route', '@require_*'], 'config': 'pyproject.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'visit_*,do_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_names': ['visit_*', 'do_*'], 'config': 'pyproject.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'make_whitelist': True, 'config': 'pyproject.toml'})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'min_confidence': 50, 'config': 'pyproject.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'sort_by_size': True, 'config': 'pyproject.toml'})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'verbose': True, 'config': 'pyproject.toml'})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'config': 'custom.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = '--exclude'
    var_2 = 'test_*'
    var_3 = '--min-confidence'
    var_4 = '75'
    var_5 = '--verbose'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._parse_args(var_6)
    var_8 = bool(var_7 == {'paths': ['file.py'], 'exclude': ['test_*'], 'min_confidence': 75, 'verbose': True, 'config': 'pyproject.toml'})
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0._check_output_config(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_config_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_verbose_toml_path. Retrieved 1/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['path1', 'path2'])
    assert var_5 is True
    var_6 = var_3['exclude']
    var_7 = bool(var_3['exclude'] == [])
    assert var_7 is True
    var_8 = var_3['ignore_decorators']
    var_9 = bool(var_3['ignore_decorators'] == [])
    assert var_9 is True
    var_10 = var_3['ignore_names']
    var_11 = bool(var_3['ignore_names'] == [])
    assert var_11 is True
    var_12 = var_3['make_whitelist']
    assert var_12 is False
    var_13 = var_3['min_confidence']
    assert var_13 == 60
    var_14 = var_3['sort_by_size']
    assert var_14 is False
    var_15 = var_3['verbose']
    assert var_15 is False

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    exclude = ["file*.py"]\n    ignore_decorators = ["deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    exclude = ["file*.py"]\n    '
    var_1 = 'path3'
    var_2 = '--exclude'
    var_3 = 'dir/'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    verbose = true\n    '



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_verbose_output_when_toml_detected. Retrieved 5/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true\n'
    var_1 = '--config'
    var_2 = 'pyproject.toml'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['verbose']
    assert var_5 is True
    var_6 = 'Reading configuration from pyproject.toml'



# Parsed testcases at query #6
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 1
    var_4 = 'value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 2
    var_8 = 'another'
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #8
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 'default'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 4/14 statements.


def test_case_0():
    var_0 = '[vulture]\nmin_confidence = 0.5\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False



# Parsed testcases at query #11
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['verbose']
    assert var_4 is True
    var_5 = var_3['config']
    assert var_5 == 'test.toml'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_verbose_output_when_toml_detected. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'valid_toml_file.toml'



# Parsed testcases at query #15
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'unknown_key'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #16
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2, var_0)
    var_4 = var_3['verbose']
    assert var_4 is True
    var_5 = var_3['config']
    assert var_5 == 'test.toml'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test.toml'



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_predicate_at_line_25_evaluates_to_true.




# Parsed testcases at query #19
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'test.toml'
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2, var_0)
    var_4 = var_3['verbose']
    assert var_4 is True
    var_5 = var_3['config']
    assert var_5 == 'test.toml'



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #21
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 42
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = False
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test.toml'



# Parsed testcases at query #23
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['verbose']
    assert var_4 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_unknown_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_wrong_type. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['verbose']
    assert var_7 is True
    var_8 = var_6['min_confidence']
    assert var_8 == 50
    var_9 = var_6['paths']
    var_10 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_10 is True

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        verbose = true\n        min_confidence = 50\n        paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        verbose = false\n        min_confidence = 50\n    '
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '70'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['verbose']
    assert var_2 is False
    var_3 = var_1['min_confidence']
    assert var_3 == 60
    var_4 = var_1['paths']
    var_5 = bool(var_1['paths'] == [])
    assert var_5 is True

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        unknown_key = "value"\n    '
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Data type for min_confidence must be'

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 1
    var_4 = 'value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 0
    var_8 = ''
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_with_tomlfile. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_verbose_toml_path. Retrieved 1/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = '--verbose'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['min_confidence']
    assert var_7 == 80
    var_8 = var_6['verbose']
    assert var_8 is True
    var_9 = var_6['paths']
    var_10 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_10 is True

def test_case_0():
    var_0 = b'\n        [tool.vulture]\n        min_confidence = 90\n        verbose = true\n        paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = b'\n        [tool.vulture]\n        min_confidence = 90\n        verbose = false\n    '
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['min_confidence']
    var_3 = var_1['verbose']
    var_4 = var_1['paths']

def test_case_0():
    var_0 = b'\n        [tool.vulture]\n        verbose = true\n    '



