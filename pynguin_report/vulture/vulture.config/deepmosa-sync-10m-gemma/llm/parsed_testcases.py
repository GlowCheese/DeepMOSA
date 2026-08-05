####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/11 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/10 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 6/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_bool. Retrieved 6/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'debug'
    var_2 = 'name'
    var_3 = 10
    var_4 = False
    var_5 = 'app'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = True
    var_9 = 'test'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'unknown'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)
    var_7 = 'Unknown configuration key: unknown'

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '5'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for timeout must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for timeout must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for debug must be 'bool'"



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = '/tmp/test'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0._check_output_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0._check_output_config(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 20/27 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 41/51 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 18/25 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'exclude'
    var_4 = 'ignore_decorators'
    var_5 = 'ignore_names'
    var_6 = 'make_whitelist'
    var_7 = 'min_confidence'
    var_8 = 'sort_by_size'
    var_9 = 'test_path'
    var_10 = [var_9]
    var_11 = 'pyproject.toml'
    var_12 = False
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 100
    var_17 = 'test_path'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = var_19['paths']
    var_21 = bool(var_19['paths'] == ['test_path'])
    assert var_21 is True
    var_22 = var_19['verbose']
    assert var_22 is False

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = 'exclude'
    var_5 = 'ignore_decorators'
    var_6 = 'ignore_names'
    var_7 = 'make_whitelist'
    var_8 = 'min_confidence'
    var_9 = 'sort_by_size'
    var_10 = 'cli_path'
    var_11 = [var_10]
    var_12 = 'pyproject.toml'
    var_13 = True
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = 100
    var_19 = 'min_confidence'
    var_20 = 'verbose'
    var_21 = 'paths'
    var_22 = 50
    var_23 = True
    var_24 = 'toml_path'
    var_25 = [var_24]
    var_26 = 'config'
    var_27 = 'exclude'
    var_28 = 'ignore_decorators'
    var_29 = 'ignore_names'
    var_30 = 'make_whitelist'
    var_31 = 'sort_by_size'
    var_32 = []
    var_33 = False
    var_34 = 'pyproject.toml'
    var_35 = []
    var_36 = []
    var_37 = []
    var_38 = {var_21: var_32, var_20: var_33, var_26: var_34, var_27: var_35, var_28: var_36, var_29: var_37, var_30: var_33, var_19: var_33, var_31: var_33}
    var_39 = 'cli_path'
    var_40 = [var_39]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'exclude'
    var_4 = 'ignore_decorators'
    var_5 = 'ignore_names'
    var_6 = 'make_whitelist'
    var_7 = 'min_confidence'
    var_8 = 'sort_by_size'
    var_9 = []
    var_10 = 'pyproject.toml'
    var_11 = False
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 100
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 9/18 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'vulture'
    var_2 = '--verbose'
    var_3 = [var_1, var_2]
    var_4 = 'verbose'
    var_5 = 'config'
    var_6 = True
    var_7 = 'dummy.toml'
    var_8 = b'verbose = true'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 7/16 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'verbose'
    var_2 = 'config'
    var_3 = True
    var_4 = 'dummy'
    var_5 = '--verbose'
    var_6 = [var_5]



# Parsed testcases at query #6
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 10
    var_4 = True
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 20
    var_8 = False
    var_9 = 'new_service'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'unsupported_key'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '20'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 10
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5
    var_6 = False
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_7)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_parse_toml_valid_config. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_empty_vulture_section. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_missing_tool_section. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_unknown_key_raises_error. Retrieved 1/6 statements.
# Partially parsed test_parse_toml_wrong_type_raises_error. Retrieved 1/4 statements.
# Partially parsed test_parse_toml_type_mismatch_logic. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "defaults"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = false\npaths = ["path1", "path2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\n'

def test_case_0():
    var_0 = '\n[other_section]\nkey = "value"\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_1 = 'Unknown configuration key: unknown_key'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "string_instead_of_int"\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 5/20 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 21/24 statements.
# Partially parsed test_make_config_toml_and_cli_merge. Retrieved 17/24 statements.
# Partially parsed test_make_config_error_on_empty_paths. Retrieved 8/13 statements.
# Partially parsed test_make_config_defaults_application. Retrieved 10/15 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'exclude'
    var_4 = 'ignore_decorators'
    var_5 = 'ignore_names'
    var_6 = 'make_whitelist'
    var_7 = 'min_confidence'
    var_8 = 'sort_by_size'
    var_9 = '.'
    var_10 = [var_9]
    var_11 = 'pyproject.toml'
    var_12 = False
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 80
    var_17 = '.'
    var_18 = [var_17]
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)
    var_21 = var_20['paths']
    var_22 = bool(var_20['paths'] == ['.'])
    assert var_22 is True
    var_23 = var_20['min_confidence']
    assert var_23 == 80

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'min_confidence'
    var_4 = 'verbose'
    var_5 = '.'
    var_6 = [var_5]
    var_7 = 'pyproject.toml'
    var_8 = 90
    var_9 = False
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 50
    var_13 = True
    var_14 = '--min-confidence'
    var_15 = '90'
    var_16 = [var_14, var_15]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = []
    var_4 = 'pyproject.toml'
    var_5 = False
    var_6 = []
    var_7 = module_0.make_config(var_6)
    var_8 = bool(False)
    assert var_8 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = '.'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = False
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 8/16 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'dummy.toml'
    var_3 = True
    var_4 = '--verbose'
    var_5 = [var_4]
    var_6 = b'dummy content'
    var_7 = module_0.make_config(var_5, var_3)
    var_8 = var_7['verbose']
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_parse_args_exclude. Retrieved 3/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/file.py'
    var_1 = 'another/dir'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': ['path/to/file.py', 'another/dir']})
    assert var_4 is True

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'pattern1,pattern2'
    var_2 = [var_0, var_1]

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@decorator1,@decorator2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_decorators': ['@decorator1', '@decorator2']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'name1,name2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_names': ['name1', 'name2']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'make_whitelist': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'min_confidence': 50})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'sort_by_size': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom_config.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'config': 'custom_config.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'verbose': True})
    assert var_3 is True
    var_4 = '--verbose'
    var_5 = [var_4]
    var_6 = module_0._parse_args(var_5)
    var_7 = bool(var_6 == {'verbose': True})
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/dir'
    var_1 = '--exclude'
    var_2 = 'test.py'
    var_3 = '--min-confidence'
    var_4 = '10'
    var_5 = '--verbose'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._parse_args(var_6)
    var_8 = bool(var_7 == {'paths': ['path/to/dir'], 'exclude': ['test.py'], 'min_confidence': 10, 'verbose': True})
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'verbose'
    var_1 = 'config'
    var_2 = True
    var_3 = 'dummy'
    var_4 = b'data'
    var_5 = '/path/to/toml'
    var_6 = '--verbose'
    var_7 = [var_6]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_on_disk. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'verbose = true\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = '--config'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 5/21 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = '--config'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 10/14 statements.
# Partially parsed test_check_input_config_wrong_type_string_to_int. Retrieved 9/13 statements.
# Partially parsed test_check_input_config_bool_vs_int_mismatch. Retrieved 8/12 statements.
# Partially parsed test_check_input_config_int_vs_bool_mismatch. Retrieved 9/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'host'
    var_1 = 'port'
    var_2 = 'debug'
    var_3 = 'localhost'
    var_4 = 8080
    var_5 = False
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = '127.0.0.1'
    var_8 = 9000
    var_9 = True
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'host'
    var_1 = 'localhost'
    var_2 = {var_0: var_1}
    var_3 = 'unknown'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)
    var_7 = 'Should have raised InputError for unknown key'
    var_8 = AssertionError(var_7)
    var_9 = 0
    var_10 = 'Unknown configuration key'

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = '8080'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError for type mismatch'
    var_7 = AssertionError(var_6)
    var_8 = 0
    var_9 = "Data type for port must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError because bool != int'
    var_7 = AssertionError(var_6)
    var_8 = "Data type for debug must be 'bool'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 80
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError because bool != int'
    var_7 = AssertionError(var_6)
    var_8 = 0
    var_9 = "Data type for port must be 'int'"



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/11 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/10 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 5/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 10
    var_4 = True
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 20
    var_8 = False
    var_9 = 'new_service'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'invalid_key'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)
    var_7 = 'Unknown configuration key: invalid_key'

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for timeout must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)
    var_5 = "Data type for enabled must be 'bool'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)



# Parsed testcases at query #17
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_evaluates_toml_path_is_file_true. Retrieved 2/12 statements.


def test_case_0():
    var_0 = b''
    var_1 = '--config'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 22/27 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 17/24 statements.
# Partially parsed test_make_config_raises_input_error_on_empty_paths. Retrieved 8/13 statements.
# Partially parsed test_make_config_detects_toml_file_from_path. Retrieved 12/20 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'exclude'
    var_4 = 'ignore_decorators'
    var_5 = 'ignore_names'
    var_6 = 'make_whitelist'
    var_7 = 'min_confidence'
    var_8 = 'sort_by_size'
    var_9 = 'test_dir'
    var_10 = [var_9]
    var_11 = 'pyproject.toml'
    var_12 = False
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 100
    var_17 = True
    var_18 = '--sort-by-size'
    var_19 = 'test_dir'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = var_21['sort_by_size']
    assert var_22 is True
    var_23 = var_21['paths']
    var_24 = bool(var_21['paths'] == ['test_dir'])
    assert var_24 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 50\npaths = ["toml_path"]'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = 'min_confidence'
    var_5 = 'cli_path'
    var_6 = [var_5]
    var_7 = 'pyproject.toml'
    var_8 = True
    var_9 = 80
    var_10 = 50
    var_11 = 'toml_path'
    var_12 = [var_11]
    var_13 = '--min-confidence'
    var_14 = '80'
    var_15 = 'cli_path'
    var_16 = [var_13, var_14, var_15]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = []
    var_4 = 'pyproject.toml'
    var_5 = False
    var_6 = []
    var_7 = module_0.make_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'existing_pyproject.toml'
    var_6 = False
    var_7 = 'from_toml'
    var_8 = [var_7]
    var_9 = 'test'
    var_10 = [var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = var_11['paths']
    var_13 = bool(var_11['paths'] == ['from_toml'])
    assert var_13 is True



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 10
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '10'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #21
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 10
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '10'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_at_cli_path. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'verbose = true'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = '--config'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = b''
    var_1 = '/fake/path/pyproject.toml'
    var_2 = []



# Parsed testcases at query #24
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 'string'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = 'hello'
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'unknown'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'not_an_int'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'True'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_via_cli. Retrieved 8/25 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'verbose = true'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = True
    var_5 = '--config'
    var_6 = [var_5, var_3]
    var_7 = module_0.make_config(var_6)
    var_8 = var_7['verbose']
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'dummy.toml'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = b''
    var_6 = '/path/to/fake.toml'
    var_7 = '--verbose'
    var_8 = [var_7]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_make_config_merges_cli_and_toml. Retrieved 19/27 statements.


def test_case_0():
    var_0 = 'exclude'
    var_1 = 'ignore_decorators'
    var_2 = 'ignore_names'
    var_3 = 'make_whitelist'
    var_4 = 'min_confidence'
    var_5 = 'sort_by_size'
    var_6 = 'config'
    var_7 = 'verbose'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = False
    var_12 = 80
    var_13 = 'pyproject.toml'
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_11, var_6: var_13, var_7: var_11}
    var_15 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    var_16 = '--min-confidence'
    var_17 = '90'
    var_18 = [var_16, var_17]

import vulture.config as module_0

def test_case_0():
    var_0 = 'exclude'
    var_1 = 'ignore_decorators'
    var_2 = 'ignore_names'
    var_3 = 'make_whitelist'
    var_4 = 'min_confidence'
    var_5 = 'sort_by_size'
    var_6 = 'config'
    var_7 = 'verbose'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = False
    var_12 = 80
    var_13 = 'pyproject.toml'
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_11, var_6: var_13, var_7: var_11}
    var_15 = 'test.py'
    var_16 = [var_15]
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = var_18['min_confidence']
    assert var_19 == 80
    var_20 = var_18['exclude']
    var_21 = bool(var_18['exclude'] == [])
    assert var_21 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'exclude'
    var_1 = 'ignore_decorators'
    var_2 = 'ignore_names'
    var_3 = 'make_whitelist'
    var_4 = 'min_confidence'
    var_5 = 'sort_by_size'
    var_6 = 'config'
    var_7 = 'verbose'
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = False
    var_12 = 80
    var_13 = 'pyproject.toml'
    var_14 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_11, var_6: var_13, var_7: var_11}
    var_15 = []
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/11 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/10 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 6/10 statements.
# Partially parsed test_check_input_config_wrong_type_str_to_bool. Retrieved 6/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 'debug'
    var_2 = 'name'
    var_3 = 8080
    var_4 = False
    var_5 = 'server'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 9000
    var_8 = True
    var_9 = 'client'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = 'invalid_key'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)
    var_7 = 'Unknown configuration key: invalid_key'

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = '8080'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for port must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for port must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'False'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for debug must be 'bool'"



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_make_config_evaluates_predicate_true_when_toml_path_is_file. Retrieved 8/24 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = b"key = 'value'"
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = '--config'
    var_7 = module_0.make_config(var_2)
    var_8 = var_7['some_key']
    assert var_8 == 'some_value'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/file.py'
    var_1 = 'another/dir'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['path/to/file.py', 'another/dir'])
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py,venv'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['exclude']
    var_5 = bool(var_3['exclude'] == ['test_*.py', 'venv'])
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@route,@auth'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['ignore_decorators']
    var_5 = bool(var_3['ignore_decorators'] == ['@route', '@auth'])
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['min_confidence']
    assert var_4 == 80

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['make_whitelist']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['sort_by_size']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['verbose']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = 'config'
    var_3 = bool('config' not in var_1)
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['config']
    assert var_4 == 'custom.toml'

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '-v'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = var_5['paths']
    var_7 = bool(var_5['paths'] == ['path/to/code'])
    assert var_7 is True
    var_8 = var_5['min_confidence']
    assert var_8 == 50
    var_9 = var_5['verbose']
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_toml_valid_config. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_empty_vulture_section. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_missing_tool_section. Retrieved 1/3 statements.
# Partially parsed test_parse_toml_unknown_key_raises_error. Retrieved 1/6 statements.
# Partially parsed test_parse_toml_wrong_type_raises_error. Retrieved 1/6 statements.
# Partially parsed test_parse_toml_bool_as_int_raises_error. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\n'

def test_case_0():
    var_0 = '\n[other_section]\nkey = "value"\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nunknown_key = 123\n'
    var_1 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_1 = "Data type for min_confidence must be 'int'"

def test_case_0():
    var_0 = '\n[tool.vulture]\nverbose = 1\n'
    var_1 = "Data type for verbose must be 'bool'"



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 4/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = var_2['verbose']
    assert var_3 is False
    var_4 = var_2['paths']
    var_5 = bool(var_2['paths'] == ['test_path'])
    assert var_5 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 50'
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'verbose'
    var_1 = True
    var_2 = b'content'
    var_3 = '--verbose'
    var_4 = [var_3]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 4/17 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = '--config'
    var_3 = None



# Parsed testcases at query #6
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 'debug'
    var_2 = 'name'
    var_3 = 8080
    var_4 = False
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 9000
    var_8 = True
    var_9 = 'new_service'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = 'invalid_key'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = '8080'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'test.toml'
    var_3 = True
    var_4 = b'some data'
    var_5 = '--verbose'
    var_6 = [var_5]



# Parsed testcases at query #8
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'threshold'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '10'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_precedence. Retrieved 5/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = 'some_path.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['min_confidence']
    assert var_5 == 50
    var_6 = var_4['paths']
    var_7 = bool(var_4['paths'] == ['some_path.py'])
    assert var_7 is True
    var_8 = 'sort_by_size'
    var_9 = bool('sort_by_size' in var_4)
    assert var_9 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 10\nverbose = true'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path/to/dir'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'nonexistent.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = 'config'
    var_4 = bool('config' in var_2)
    assert var_4 is True
    var_5 = var_2['config']
    assert var_5 == 'pyproject.toml'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 15/22 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = '--verbose'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['test_file.py'])
    assert var_5 is True
    var_6 = var_3['verbose']
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = false'
    var_1 = 'min_confidence'
    var_2 = 'verbose'
    var_3 = 'paths'
    var_4 = 'config'
    var_5 = 10
    var_6 = False
    var_7 = 'default_path'
    var_8 = [var_7]
    var_9 = 'pyproject.toml'
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_8, var_4: var_9}
    var_11 = 'extra_path.py'
    var_12 = '--verbose'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13, var_11)
    var_15 = var_14['verbose']
    assert var_15 is True
    var_16 = var_14['min_confidence']
    assert var_16 == 50
    var_17 = 'extra_path.py'
    var_18 = bool('extra_path.py' in var_14['paths'])
    assert var_18 is True

import vulture.config as module_0

def test_case_0():
    var_0 = ''
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)



# Parsed testcases at query #11
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '10'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 9/18 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'fake_path.toml'
    var_3 = True
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = '--verbose'
    var_7 = [var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = var_8['verbose']
    assert var_9 is True
    var_10 = 'fake_path.toml'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = b'dummy content'
    var_1 = '--verbose'
    var_2 = [var_1]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_on_disk. Retrieved 8/23 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[version = "1.0"]'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = False
    var_5 = '--config'
    var_6 = [var_5, var_0]
    var_7 = module_0.make_config(var_6)
    var_8 = var_7['version']
    assert var_8 == '1.0'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 5/22 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'



# Parsed testcases at query #16
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '10'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 37/56 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'exclude'
    var_2 = 'ignore_decorators'
    var_3 = 'ignore_names'
    var_4 = 'make_whitelist'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'config'
    var_8 = 'verbose'
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = False
    var_14 = 'pyproject.toml'
    var_15 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_13, var_6: var_13, var_7: var_14, var_8: var_13}
    var_16 = '.'
    var_17 = 'paths'
    var_18 = 'exclude'
    var_19 = 'ignore_decorators'
    var_20 = 'ignore_names'
    var_21 = 'make_whitelist'
    var_22 = 'min_confidence'
    var_23 = 'sort_by_size'
    var_24 = 'config'
    var_25 = 'verbose'
    var_26 = '.'
    var_27 = [var_26]
    var_28 = []
    var_29 = []
    var_30 = []
    var_31 = False
    var_32 = 'pyproject.toml'
    var_33 = '.'
    var_34 = [var_33]
    var_35 = None
    var_36 = module_0.make_config(var_34, var_35)
    var_37 = var_36['paths']
    var_38 = bool(var_36['paths'] == ['.'])
    assert var_38 is True
    var_39 = var_36['verbose']
    assert var_39 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_at_cli_path. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'verbose = true\n'
    var_1 = '--config'



# Parsed testcases at query #19
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_25. Retrieved 6/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'existing_file.toml'
    var_1 = '--config'
    var_2 = 'existing_file.toml'
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = b''
    var_1 = '/fake/path.toml'
    var_2 = '--verbose'
    var_3 = [var_2]



# Parsed testcases at query #22
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 8080
    var_2 = {var_0: var_1}
    var_3 = '8080'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #23
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'verbose'
    var_2 = 'name'
    var_3 = 10
    var_4 = True
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = False
    var_9 = 'new_service'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'unknown_key'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '5'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'verbose'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_at_cli_path. Retrieved 5/23 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = '--config'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = b''



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_25. Retrieved 8/18 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_config.toml'
    var_1 = "some = 'data'"
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = False
    var_5 = []
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 5/8 statements.
# Partially parsed test_check_input_config_wrong_type_str_to_bool. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 30
    var_4 = True
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 60
    var_8 = False
    var_9 = 'new_service'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = 'invalid_key'
    var_4 = 10
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_5)
    var_7 = 'Unknown configuration key: invalid_key'

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = '60'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for timeout must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)
    var_5 = "Data type for enabled must be 'bool'"

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'True'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for enabled must be 'bool'"



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 7/11 statements.
# Partially parsed test_make_config_toml_and_cli_merge. Retrieved 11/17 statements.
# Partially parsed test_make_config_raises_input_error_on_empty_paths. Retrieved 6/10 statements.
# Partially parsed test_make_config_reads_existing_toml_file. Retrieved 10/17 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'pyproject.toml'
    var_5 = [var_2]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['.'])
    assert var_8 is True
    var_9 = 'verbose'
    var_10 = bool('verbose' in var_6)
    assert var_10 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 50\n'
    var_1 = 'paths'
    var_2 = 'min_confidence'
    var_3 = '.'
    var_4 = [var_3]
    var_5 = 80
    var_6 = 'verbose'
    var_7 = True
    var_8 = 50
    var_9 = []
    var_10 = [var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = []
    var_3 = 'pyproject.toml'
    var_4 = []
    var_5 = module_0.make_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'pyproject.toml'
    var_5 = 'verbose'
    var_6 = [var_2]
    var_7 = False
    var_8 = [var_2]
    var_9 = module_0.make_config(var_8)
    var_10 = var_9['paths']
    var_11 = bool(var_9['paths'] == ['.'])
    assert var_11 is True



