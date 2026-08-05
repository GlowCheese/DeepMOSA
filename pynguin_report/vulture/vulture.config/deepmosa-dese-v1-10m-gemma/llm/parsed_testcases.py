####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_config_with_empty_args_and_no_toml. Retrieved 7/13 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 19/27 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 6/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = 'pyproject.toml'
    var_2 = False
    var_3 = None
    var_4 = 'test_path'
    var_5 = []
    var_6 = module_0.make_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'cli_path'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = False
    var_4 = True
    var_5 = None
    var_6 = 'tool'
    var_7 = 'vulture'
    var_8 = 'paths'
    var_9 = 'sort_by_size'
    var_10 = 'toml_path'
    var_11 = [var_10]
    var_12 = {var_8: var_11, var_9: var_3}
    var_13 = {var_7: var_12}
    var_14 = {var_6: var_13}
    var_15 = '--sort-by-size'
    var_16 = [var_15]
    var_17 = ''
    var_18 = module_0.make_config(var_16, var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = 'pyproject.toml'
    var_2 = False
    var_3 = None
    var_4 = []
    var_5 = module_0.make_config(var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_at_cli_path. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'config.toml'
    var_1 = 'verbose = true'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = True
    var_5 = '--config'
    var_6 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_bool. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 'debug'
    var_2 = 'name'
    var_3 = 8080
    var_4 = False
    var_5 = 'server'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 90
    var_8 = True
    var_9 = 'test'
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = '--verbose'
    var_2 = [var_1]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_parse_args_exclude. Retrieved 3/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/dir'
    var_1 = 'file.py'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

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

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'name1,name2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom_config.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = '--verbose'
    var_4 = [var_3]
    var_5 = module_0._parse_args(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = '--sort-by-size'
    var_4 = '-v'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0._parse_args(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 7/16 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'verbose'
    var_2 = 'config'
    var_3 = True
    var_4 = '/tmp/fake_config.toml'
    var_5 = '--verbose'
    var_6 = [var_5]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 10/15 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 16/26 statements.
# Partially parsed test_make_config_raises_input_error_on_empty_paths. Retrieved 10/17 statements.
# Partially parsed test_make_config_uses_default_toml_path_if_file_exists. Retrieved 13/23 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = False
    var_7 = 'test_path'
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 20\nverbose = true'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'min_confidence'
    var_4 = 'verbose'
    var_5 = 'cli_path'
    var_6 = [var_5]
    var_7 = 'pyproject.toml'
    var_8 = 50
    var_9 = True
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 20
    var_13 = True
    var_14 = 'cli_path'
    var_15 = [var_14]

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
    var_8 = 'Should have raised InputError'
    var_9 = AssertionError(var_8)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = False
    var_7 = 'paths'
    var_8 = 'test'
    var_9 = [var_8]
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = module_0.make_config(var_11)



# Parsed testcases at query #8
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 10
    var_4 = True
    var_5 = 'default'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 30
    var_8 = False
    var_9 = 'custom'
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
    var_7 = 'InputError not raised for unknown key'
    var_8 = AssertionError(var_7)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'InputError not raised for wrong type (int to str)'
    var_7 = AssertionError(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'InputError not raised for wrong type (int to bool)'
    var_7 = AssertionError(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = '/fake/path/pyproject.toml'
    var_2 = 'config'
    var_3 = 'verbose'
    var_4 = 'some_path'
    var_5 = True
    var_6 = '--verbose'
    var_7 = [var_6]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 4/16 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 26/38 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'min_confidence'
    var_3 = []
    var_4 = False
    var_5 = {var_0: var_3, var_1: var_4, var_2: var_4}
    var_6 = 'paths'
    var_7 = 'verbose'
    var_8 = 'config'
    var_9 = 'cli.py'
    var_10 = [var_9]
    var_11 = True
    var_12 = 'pyproject.toml'
    var_13 = {var_6: var_10, var_7: var_11, var_8: var_12}
    var_14 = 'tool'
    var_15 = 'vulture'
    var_16 = 'min_confidence'
    var_17 = 'toml.py'
    var_18 = [var_17]
    var_19 = 50
    var_20 = {var_6: var_18, var_16: var_19}
    var_21 = {var_15: var_20}
    var_22 = {var_14: var_21}
    var_23 = b'dummy'
    var_24 = 'cli.py'
    var_25 = [var_24]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'config'
    var_3 = 'default.py'
    var_4 = [var_3]
    var_5 = False
    var_6 = 'pyproject.toml'
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = []
    var_9 = module_0.make_config(var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_config_with_cli_only. Retrieved 4/23 statements.
# Failed to parse test_make_config_raises_error_on_empty_paths.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #12
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_input_config_detects_type_mismatch. Retrieved 8/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 19/38 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = 'paths'
    var_2 = 'exclude'
    var_3 = 'ignore_decorators'
    var_4 = 'ignore_names'
    var_5 = 'make_whitelist'
    var_6 = 'min_confidence'
    var_7 = 'sort_by_size'
    var_8 = 'config'
    var_9 = 'verbose'
    var_10 = [var_0]
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = False
    var_15 = 'pyproject.toml'
    var_16 = 'test_path'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #15
#--------------------------




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

import vulture.config as module_0

def test_case_0():
    var_0 = 'host'
    var_1 = 'localhost'
    var_2 = {var_0: var_1}
    var_3 = 123
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

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
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = '--verbose'
    var_2 = [var_1]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_25. Retrieved 6/25 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = None



# Parsed testcases at query #18
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 2/11 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 6/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_config_exists.toml'
    var_1 = 'verbose = true'
    var_2 = '--config'
    var_3 = [var_2, var_0]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_override. Retrieved 5/9 statements.
# Partially parsed test_make_config_error_on_empty_paths. Retrieved 4/14 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '20'
    var_2 = '--sort-by-size'
    var_3 = 'path/to/code'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 10\nverbose = false\n'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path/to/code'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = []\n'
    var_1 = '--config'
    var_2 = 'nonexistent.toml'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file_path_in_cli. Retrieved 10/27 statements.
# Partially parsed test_make_config_predicate_true. Retrieved 10/28 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'verbose = true\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = 'verbose'
    var_5 = True
    var_6 = '--config'
    var_7 = [var_6, var_3]
    var_8 = None
    var_9 = module_0.make_config(var_7, var_8)

import vulture.config as module_0

def test_case_0():
    var_0 = 'verbose = true\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = True
    var_4 = 'verbose'
    var_5 = True
    var_6 = '--config'
    var_7 = [var_6, var_5]
    var_8 = None
    var_9 = module_0.make_config(var_7, var_8)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_input_config_detects_type_mismatch. Retrieved 8/12 statements.
# Partially parsed test_check_input_config_detects_bool_vs_int_mismatch. Retrieved 6/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 17/24 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'paths'
    var_3 = 'pyproject.toml'
    var_4 = False
    var_5 = []
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = '1.0.0'
    var_8 = 'pyproject.toml'
    var_9 = 'test_path'
    var_10 = [var_9]
    var_11 = True
    var_12 = None
    var_13 = False
    var_14 = 'test_path'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'paths'
    var_3 = 'pyproject.toml'
    var_4 = False
    var_5 = []
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = b'[tool.vulture]\nmin_confidence = 20\nverbose = true'
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = [var_8, var_9]
    var_11 = module_0.make_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'paths'
    var_3 = 'pyproject.toml'
    var_4 = False
    var_5 = []
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = module_0.make_config(var_7)



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 7/24 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_2]
    var_6 = module_0.make_config(var_5)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = '/path/to/data'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0._check_output_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0._check_output_config(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_parse_toml_valid_config. Retrieved 14/18 statements.
# Partially parsed test_parse_toml_empty_section. Retrieved 3/7 statements.
# Partially parsed test_parse_toml_unknown_key_raises_error. Retrieved 3/9 statements.
# Partially parsed test_parse_toml_wrong_type_raises_error. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\n'
    var_1 = 'exclude'
    var_2 = 'ignore_decorators'
    var_3 = 'ignore_names'
    var_4 = 'make_whitelist'
    var_5 = 'min_confidence'
    var_6 = 'sort_by_size'
    var_7 = 'verbose'
    var_8 = 'paths'
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = False
    var_13 = []

def test_case_0():
    var_0 = 'verbose'
    var_1 = False
    var_2 = '\n[tool.vulture]\n'

def test_case_0():
    var_0 = 'verbose'
    var_1 = False
    var_2 = '\n[tool.vulture]\nunknown_key = True\n'

def test_case_0():
    var_0 = 'min_confidence'
    var_1 = 0
    var_2 = '\n[tool.vulture]\nmin_confidence = "high"\n'



# Parsed testcases at query #3
#--------------------------




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
    var_9 = 'prod'
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
    var_3 = 'False'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_input_config_detects_type_mismatch. Retrieved 8/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #5
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test.py,venv'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@route,@auth'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'temp_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = '--exclude'
    var_2 = 'pattern1'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '-v'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0._parse_args(var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_config_merges_cli_and_defaults. Retrieved 4/20 statements.
# Partially parsed test_make_config_toml_precedence. Retrieved 10/16 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 3/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 50'
    var_1 = 'config'
    var_2 = 'pyproject.toml'
    var_3 = {var_1: var_2}
    var_4 = 'verbose'
    var_5 = 'min_confidence'
    var_6 = True
    var_7 = 50
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = []

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = str(var_0)



# Parsed testcases at query #7
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 30
    var_4 = True
    var_5 = 'server'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = False
    var_9 = 'client'
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'dummy.toml'
    var_3 = True
    var_4 = '/fake/path.toml'
    var_5 = b'verbose = true'
    var_6 = '--verbose'
    var_7 = [var_6]
    var_8 = b'content'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 6/22 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = None



# Parsed testcases at query #10
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 10
    var_4 = True
    var_5 = 'test'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = False
    var_9 = 'prod'
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
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 8/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'fake_config.toml'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_2]
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 5/17 statements.


def test_case_0():
    var_0 = b"key = 'value'"
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 4/15 statements.
# Partially parsed test_make_config_merging_toml_and_cli. Retrieved 13/30 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 2/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["*.tmp"]\nmin_confidence = 50'
    var_1 = 'path1'
    var_2 = 'tool'
    var_3 = 'vulture'
    var_4 = 'exclude'
    var_5 = 'min_confidence'
    var_6 = '*.tmp'
    var_7 = [var_6]
    var_8 = 50
    var_9 = {var_4: var_7, var_5: var_8}
    var_10 = {var_3: var_9}
    var_11 = 'path1'
    var_12 = [var_11]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/8 statements.
# Partially parsed test_make_config_with_multiple_cli_args. Retrieved 7/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'non_existent.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = false'
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = [var_1, var_2]

def test_case_0():
    pass

def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["test.py"]'
    var_1 = '--exclude'
    var_2 = 'a,b'
    var_3 = '--sort-by-size'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'pattern1,pattern2'
    var_6 = [var_1, var_5]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 5/15 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = 'verbose'
    var_4 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 5/8 statements.
# Partially parsed test_check_input_config_int_to_bool. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 'name'
    var_3 = 30
    var_4 = True
    var_5 = 'default'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 60
    var_8 = False
    var_9 = 'custom'
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = '60'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_defaults. Retrieved 7/15 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 15/24 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'pyproject.toml'
    var_3 = False
    var_4 = 'some_path'
    var_5 = [var_4]
    var_6 = module_0.make_config(var_5)

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 20\nverbose = true'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = [var_1, var_2]
    var_4 = 'min_confidence'
    var_5 = 'config'
    var_6 = 50
    var_7 = 'pyproject.toml'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'min_confidence'
    var_10 = 'verbose'
    var_11 = 'config'
    var_12 = 0
    var_13 = False
    var_14 = 'pyproject.toml'

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_defaults. Retrieved 5/8 statements.
# Partially parsed test_make_config_cli_precedence. Retrieved 11/16 statements.
# Partially parsed test_make_config_with_toml_file. Retrieved 10/15 statements.
# Partially parsed test_make_config_raises_input_error_on_empty_paths. Retrieved 7/10 statements.
# Partially parsed test_make_config_reads_existing_toml_from_disk. Retrieved 10/14 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'pyproject.toml'
    var_2 = ''
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'pyproject.toml'
    var_3 = True
    var_4 = 'paths'
    var_5 = False
    var_6 = 'test.py'
    var_7 = [var_6]
    var_8 = '--verbose'
    var_9 = [var_8]
    var_10 = b''

def test_case_0():
    var_0 = 'config'
    var_1 = 'pyproject.toml'
    var_2 = 'paths'
    var_3 = 'exclude'
    var_4 = 'path1'
    var_5 = [var_4]
    var_6 = '*.py'
    var_7 = [var_6]
    var_8 = b'dummy'
    var_9 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'paths'
    var_2 = 'pyproject.toml'
    var_3 = []
    var_4 = ''
    var_5 = [var_4]
    var_6 = module_0.make_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'pyproject.toml'
    var_2 = 'paths'
    var_3 = 'verbose'
    var_4 = 'dir'
    var_5 = [var_4]
    var_6 = False
    var_7 = ''
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)



# Parsed testcases at query #21
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 'enabled'
    var_2 = 10
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_3}
    var_6 = module_0._check_input_config(var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 7/17 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'dummy'
    var_4 = True
    var_5 = '--verbose'
    var_6 = [var_5]



# Parsed testcases at query #23
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 30
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_3}
    var_6 = module_0._check_input_config(var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_with_toml_overriding. Retrieved 19/25 statements.
# Partially parsed test_make_config_cli_precedence. Retrieved 23/27 statements.


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
    var_16 = 'path/to/code'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)

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
    var_16 = b'[tool.vulture]\nexclude = ["*.tmp"]\nverbose = true'
    var_17 = '.'
    var_18 = [var_17]

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
    var_16 = b'[tool.vulture]\nmin_confidence = 50\nexclude = ["old"]'
    var_17 = '--min-confidence'
    var_18 = '80'
    var_19 = '--exclude'
    var_20 = 'new'
    var_21 = '.'
    var_22 = [var_17, var_18, var_19, var_20, var_21]

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
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 4/20 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 26/44 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 4/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'my_path'
    var_1 = 'my_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 50'
    var_1 = 'my_path'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'my_path'
    var_6 = 'paths'
    var_7 = 'config'
    var_8 = 'verbose'
    var_9 = 'exclude'
    var_10 = 'ignore_decorators'
    var_11 = 'ignore_names'
    var_12 = 'make_whitelist'
    var_13 = 'min_confidence'
    var_14 = 'sort_by_size'
    var_15 = [var_5]
    var_16 = 'pyproject.toml'
    var_17 = False
    var_18 = None
    var_19 = 80
    var_20 = 'tool'
    var_21 = 'vulture'
    var_22 = True
    var_23 = 50
    var_24 = {var_8: var_22, var_13: var_23}
    var_25 = {var_21: var_24}

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = []
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 9/17 statements.


def test_case_0():
    var_0 = b'verbose = true'
    var_1 = 'vulture'
    var_2 = '--verbose'
    var_3 = [var_1, var_2]
    var_4 = 'verbose'
    var_5 = 'config'
    var_6 = True
    var_7 = 'dummy'
    var_8 = b'verbose = true'



# Parsed testcases at query #27
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'enabled'
    var_2 = 10
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '30'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



