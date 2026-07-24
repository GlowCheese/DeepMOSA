####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_config_defaults. Retrieved 5/10 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 10/18 statements.
# Partially parsed test_make_config_empty_cli_uses_toml_values. Retrieved 6/13 statements.
# Partially parsed test_make_config_raises_on_invalid_output_config. Retrieved 4/12 statements.
# Partially parsed test_make_config_with_file_loading. Retrieved 6/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'pyproject.toml'
    var_2 = '--some-arg'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['verbose']
    assert var_5 is False

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\nmin_confidence = 20'
    var_1 = 'verbose'
    var_2 = 'min_confidence'
    var_3 = True
    var_4 = 50
    var_5 = 20
    var_6 = '--verbose'
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = [var_6, var_7, var_8]

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true'
    var_1 = 'config'
    var_2 = 'pyproject.toml'
    var_3 = 'verbose'
    var_4 = True
    var_5 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = []
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'pyproject.toml'
    var_2 = 'verbose'
    var_3 = False
    var_4 = []
    var_5 = module_0.make_config(var_4)
    var_6 = 'config'
    var_7 = bool('config' in var_5)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['verbose']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['verbose']
    assert var_2 is False



# Parsed testcases at query #3
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
    var_4 = bool(var_3 == {'paths': ['path/to/file.py', 'another/dir']})
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = True
    var_5 = 'string'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = False
    var_9 = 'hello'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'unknown'
    var_4 = {var_3: var_1}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError for unknown key'
    var_7 = AssertionError(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 1.5
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError for wrong type'
    var_7 = AssertionError(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError because bool is not int'
    var_7 = AssertionError(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = '1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = 'Should have raised InputError for wrong type'
    var_7 = AssertionError(var_6)



# Parsed testcases at query #5
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = '30'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_1}
    var_6 = 'not_an_int'
    var_7 = {var_0: var_6}
    var_8 = module_0._check_input_config(var_7)



# Parsed testcases at query #6
#--------------------------




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
    var_3 = 'unknown_key'
    var_4 = 123
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
    var_0 = 'threshold'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #7
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
    var_1 = 'test.py,venv/*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['exclude']
    var_5 = bool(var_3['exclude'] == ['test.py', 'venv/*'])
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
    var_0 = '--ignore-names'
    var_1 = 'func1,func2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['ignore_names']
    var_5 = bool(var_3['ignore_names'] == ['func1', 'func2'])
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['make_whitelist']
    assert var_3 is True

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
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['sort_by_size']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom_config.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3['config']
    assert var_4 == 'custom_config.toml'

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = var_2['verbose']
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--make-whitelist'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = var_5['paths']
    var_7 = bool(var_5['paths'] == ['path1'])
    assert var_7 is True
    var_8 = var_5['min_confidence']
    assert var_8 == 50
    var_9 = var_5['make_whitelist']
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 8/14 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 13/24 statements.
# Partially parsed test_make_config_raises_error_on_invalid_output. Retrieved 6/12 statements.
# Partially parsed test_make_config_with_toml_file_overridden_by_cli. Retrieved 10/20 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'test.toml'
    var_3 = True
    var_4 = '--config'
    var_5 = 'test.toml'
    var_6 = [var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = var_7['config']
    assert var_8 == 'test.toml'
    var_9 = var_7['verbose']
    assert var_9 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = false'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'cli_path'
    var_4 = [var_3]
    var_5 = 'test.toml'
    var_6 = 'min_confidence'
    var_7 = 'verbose'
    var_8 = 50
    var_9 = False
    var_10 = '--paths'
    var_11 = 'cli_path'
    var_12 = [var_10, var_11]

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'paths'
    var_2 = 'pyproject.toml'
    var_3 = []
    var_4 = []
    var_5 = module_0.make_config(var_4)
    var_6 = 'Please pass at least one file or directory'

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50'
    var_1 = 'min_confidence'
    var_2 = 'config'
    var_3 = 80
    var_4 = 'test.toml'
    var_5 = 'min_confidence'
    var_6 = 50
    var_7 = '--min-confidence'
    var_8 = '80'
    var_9 = [var_7, var_8]



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_config_with_cli_args_only. Retrieved 10/15 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 16/26 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'config'
    var_3 = 'test.py'
    var_4 = [var_3]
    var_5 = True
    var_6 = 'pyproject.toml'
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = var_9['paths']
    var_11 = bool(var_9['paths'] == ['test.py'])
    assert var_11 is True

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'cli_path.py'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = True
    var_7 = {var_0: var_4, var_1: var_5, var_2: var_6}
    var_8 = b'[tool.vulture]\nexclude = ["*.tmp"]\nmin_confidence = 50'
    var_9 = 'exclude'
    var_10 = 'min_confidence'
    var_11 = '*.tmp'
    var_12 = [var_11]
    var_13 = 50
    var_14 = 'cli_path.py'
    var_15 = [var_14]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = []
    var_3 = 'pyproject.toml'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = module_0.make_config(var_5)
    var_7 = 'Please pass at least one file or directory'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_26. Retrieved 7/25 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = b'[some_key]\nvalue = 1'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_2]
    var_6 = module_0.make_config(var_5)
    var_7 = bool(var_3)
    assert var_7 is True
    var_8 = bool(var_6 is not None)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_bool_is_not_int. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_string_to_bool. Retrieved 6/9 statements.


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
    var_9 = 'proxy'
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 7/9 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 13/19 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 16/22 statements.
# Partially parsed test_make_config_raises_error_on_invalid_output. Retrieved 7/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'test_path'
    var_3 = [var_2]
    var_4 = 'pyproject.toml'
    var_5 = [var_2]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['test_path'])
    assert var_8 is True

def test_case_0():
    var_0 = '{"tool": {"vulture": {"exclude": ["*.tmp"], "verbose": false}}}'
    var_1 = 'utf-8'
    var_2 = 'paths'
    var_3 = 'config'
    var_4 = 'src'
    var_5 = [var_4]
    var_6 = 'pyproject.toml'
    var_7 = 'exclude'
    var_8 = 'verbose'
    var_9 = '*.tmp'
    var_10 = [var_9]
    var_11 = False
    var_12 = [var_4]

def test_case_0():
    var_0 = '{"tool": {"vulture": {"exclude": ["*.tmp"], "verbose": false}}}'
    var_1 = 'utf-8'
    var_2 = 'paths'
    var_3 = 'exclude'
    var_4 = 'config'
    var_5 = 'src'
    var_6 = [var_5]
    var_7 = '*.log'
    var_8 = [var_7]
    var_9 = 'pyproject.toml'
    var_10 = 'verbose'
    var_11 = '*.tmp'
    var_12 = [var_11]
    var_13 = False
    var_14 = '--exclude'
    var_15 = [var_5, var_14, var_7]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = []
    var_3 = 'pyproject.toml'
    var_4 = 'Please pass at least one file or directory'
    var_5 = [var_4]
    var_6 = {}
    var_7 = []
    var_8 = module_0.make_config(var_7)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 15/24 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['test.py'])
    assert var_4 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 20\nverbose = true'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'min_confidence'
    var_4 = 'cli_path.py'
    var_5 = [var_4]
    var_6 = 'pyproject.toml'
    var_7 = 50
    var_8 = {var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'verbose'
    var_10 = 20
    var_11 = True
    var_12 = {var_3: var_10, var_9: var_11}
    var_13 = 'cli_path.py'
    var_14 = [var_13]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_26. Retrieved 3/16 statements.


def test_case_0():
    var_0 = b''
    var_1 = '--config'
    var_2 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_evaluates_toml_path_is_file_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = b''
    var_1 = '--config'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_predicate_false_non_existent_file. Retrieved 5/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = module_0.make_config(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_toml_path_is_not_file. Retrieved 5/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #19
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'not_an_int'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_predicate_false_when_toml_path_is_not_a_file. Retrieved 7/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'non_existent_file.toml'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_2]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['config']
    assert var_7 == 'non_existent_file.toml'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 8/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'dummy.toml'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_1]
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 7/10 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 10/15 statements.
# Partially parsed test_make_config_uses_defaults. Retrieved 5/7 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 4/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = True
    var_5 = [var_2]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['test.py'])
    assert var_8 is True
    var_9 = var_6['verbose']
    assert var_9 is True

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'cli_path'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'toml_path'
    var_6 = [var_5]
    var_7 = False
    var_8 = b'dummy'
    var_9 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['min_confidence']
    assert var_5 == 80
    var_6 = var_4['verbose']
    assert var_6 is False

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = []
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_config_triggers_toml_file_detection. Retrieved 5/21 statements.


def test_case_0():
    var_0 = b'some data'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_input_config_raises_error_on_type_mismatch. Retrieved 8/12 statements.


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
    var_8 = "Data type for timeout must be 'int'"



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

# Partially parsed test_make_config_cli_only. Retrieved 13/17 statements.
# Partially parsed test_make_config_toml_and_cli_merging. Retrieved 16/22 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 8/11 statements.
# Partially parsed test_make_config_uses_defaults. Retrieved 10/14 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'min_confidence'
    var_4 = 'test_dir'
    var_5 = [var_4]
    var_6 = 'pyproject.toml'
    var_7 = False
    var_8 = 80
    var_9 = '--min-confidence'
    var_10 = '80'
    var_11 = [var_9, var_10, var_4]
    var_12 = module_0.make_config(var_11)
    var_13 = var_12['min_confidence']
    assert var_13 == 80
    var_14 = var_12['paths']
    var_15 = bool(var_12['paths'] == ['test_dir'])
    assert var_15 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'min_confidence'
    var_4 = 'verbose'
    var_5 = 'test_dir'
    var_6 = [var_5]
    var_7 = 'pyproject.toml'
    var_8 = 90
    var_9 = True
    var_10 = 'exclude'
    var_11 = 50
    var_12 = []
    var_13 = '--min-confidence'
    var_14 = '90'
    var_15 = [var_13, var_14]

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
    var_3 = '.'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = False
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = 'min_confidence'
    var_11 = bool('min_confidence' in var_9)
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_ensures_predicate_at_line_26_is_false. Retrieved 6/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_wrong_type_bool_to_int. Retrieved 5/8 statements.
# Partially parsed test_check_input_config_wrong_type_float_to_int. Retrieved 6/9 statements.


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
    var_7 = 'Unknown configuration key: invalid_key'

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
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
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = 30.5
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = "Data type for timeout must be 'int'"



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_config_predicate_true. Retrieved 8/16 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'dummy.toml'
    var_3 = True
    var_4 = 'dummy.toml'
    var_5 = '--verbose'
    var_6 = [var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = var_7['verbose']
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 10/15 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 15/22 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 8/12 statements.
# Partially parsed test_make_config_reads_from_file_system. Retrieved 12/20 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'test.py'
    var_4 = [var_3]
    var_5 = 'pyproject.toml'
    var_6 = True
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = var_9['paths']
    var_11 = bool(var_9['paths'] == ['test.py'])
    assert var_11 is True
    var_12 = var_9['min_confidence']
    assert var_12 == 0

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\npaths = ["toml_path"]'
    var_1 = 'paths'
    var_2 = 'config'
    var_3 = 'min_confidence'
    var_4 = 'verbose'
    var_5 = 'cli_path'
    var_6 = [var_5]
    var_7 = 'pyproject.toml'
    var_8 = 80
    var_9 = True
    var_10 = 50
    var_11 = 'toml_path'
    var_12 = [var_11]
    var_13 = 'cli_path'
    var_14 = [var_13]

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
    var_9 = bool(True)
    assert var_9 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = 'test.py'
    var_4 = [var_3]
    var_5 = 'existing_config.toml'
    var_6 = False
    var_7 = 'from_file'
    var_8 = [var_7]
    var_9 = 'test.py'
    var_10 = [var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = var_11['paths']
    var_13 = bool(var_11['paths'] == ['test.py'])
    assert var_13 is True



# Parsed testcases at query #7
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
    var_4 = bool(var_3 == {'paths': ['path/to/file.py', 'another/dir']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test.py,venv/*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': [], 'exclude': ['test.py', 'venv/*']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@route,@auth'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': [], 'ignore_decorators': ['@route', '@auth']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'unused_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': [], 'ignore_names': ['unused_*']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'paths': [], 'make_whitelist': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': [], 'min_confidence': 80})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'paths': [], 'sort_by_size': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': [], 'config': 'custom.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'paths': [], 'verbose': True})
    assert var_3 is True
    var_4 = '--verbose'
    var_5 = [var_4]
    var_6 = module_0._parse_args(var_5)
    var_7 = bool(var_6 == {'paths': [], 'verbose': True})
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = var_5['paths']
    var_7 = bool(var_5['paths'] == ['path1'])
    assert var_7 is True
    var_8 = var_5['min_confidence']
    assert var_8 == 50
    var_9 = var_5['sort_by_size']
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 8/17 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'dummy.toml'
    var_2 = 'verbose'
    var_3 = False
    var_4 = '--config'
    var_5 = [var_4, var_1]
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_config_evaluates_predicate_true. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'verbose = true'
    var_2 = '--config'
    var_3 = None



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 'debug'
    var_2 = 8080
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '8080'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_input_config_unknown_key. Retrieved 7/10 statements.
# Partially parsed test_check_input_config_wrong_type_int_to_str. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_bool_instead_of_int. Retrieved 6/9 statements.
# Partially parsed test_check_input_config_int_instead_of_bool. Retrieved 5/8 statements.


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
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = True
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_evaluates_toml_path_is_file_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = b''
    var_1 = 'config'
    var_2 = '--config'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_cli_only. Retrieved 12/18 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 11/19 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 7/12 statements.
# Partially parsed test_make_config_with_existing_toml_file. Retrieved 12/21 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'config'
    var_3 = '.'
    var_4 = [var_3]
    var_5 = True
    var_6 = 'pyproject.toml'
    var_7 = []
    var_8 = False
    var_9 = [var_3]
    var_10 = None
    var_11 = module_0.make_config(var_9, var_10)
    var_12 = var_11['paths']
    var_13 = bool(var_11['paths'] == ['.'])
    assert var_13 is True
    var_14 = var_11['verbose']
    assert var_14 is True

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = ["src"]\nverbose = false\n'
    var_1 = 'paths'
    var_2 = 'verbose'
    var_3 = 'cli_path'
    var_4 = [var_3]
    var_5 = True
    var_6 = 'toml_path'
    var_7 = [var_6]
    var_8 = False
    var_9 = []
    var_10 = [var_3]
    var_11 = 'config'

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = []
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = module_0.make_config(var_5)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'config'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = 'dummy.toml'
    var_5 = 'verbose'
    var_6 = 'toml_path'
    var_7 = [var_6]
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = var_11['paths']
    var_13 = bool(var_11['paths'] == ['toml_path'])
    assert var_13 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_defaults. Retrieved 13/21 statements.
# Partially parsed test_make_config_cli_override. Retrieved 12/19 statements.
# Partially parsed test_make_config_toml_integration. Retrieved 13/25 statements.
# Partially parsed test_make_config_error_on_empty_paths. Retrieved 9/18 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'paths'
    var_2 = 'pyproject.toml'
    var_3 = 'test_path'
    var_4 = [var_3]
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = lambda : var_5
    var_7 = 'verbose'
    var_8 = 'min_confidence'
    var_9 = False
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = var_12['verbose']
    assert var_13 is False
    var_14 = var_12['min_confidence']
    assert var_14 == 0

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'verbose'
    var_2 = 'paths'
    var_3 = 'pyproject.toml'
    var_4 = True
    var_5 = 'path'
    var_6 = [var_5]
    var_7 = 'min_confidence'
    var_8 = False
    var_9 = '--verbose'
    var_10 = [var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = var_11['verbose']
    assert var_12 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true'
    var_1 = 'config'
    var_2 = 'fake.toml'
    var_3 = 'min_confidence'
    var_4 = 'verbose'
    var_5 = 50
    var_6 = True
    var_7 = 'paths'
    var_8 = 0
    var_9 = False
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'paths'
    var_2 = 'pyproject.toml'
    var_3 = []
    var_4 = 'verbose'
    var_5 = False
    var_6 = []
    var_7 = []
    var_8 = module_0.make_config(var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_valid_cli_only. Retrieved 8/14 statements.
# Partially parsed test_make_config_merges_toml_and_cli. Retrieved 20/30 statements.
# Partially parsed test_make_config_raises_error_on_empty_paths. Retrieved 11/17 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'test.py'
    var_6 = [var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = var_7['paths']
    var_9 = bool(var_7['paths'] == ['test.py'])
    assert var_9 is True
    var_10 = var_7['verbose']
    assert var_10 is True

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = 'exclude'
    var_3 = []
    var_4 = False
    var_5 = []
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = b'[tool.vulture]\nexclude = ["*.tmp"]\nverbose = false'
    var_8 = 'paths'
    var_9 = 'verbose'
    var_10 = 'src'
    var_11 = [var_10]
    var_12 = True
    var_13 = 'exclude'
    var_14 = 'verbose'
    var_15 = '*.tmp'
    var_16 = [var_15]
    var_17 = False
    var_18 = 'src'
    var_19 = [var_18]

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'verbose'
    var_2 = []
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'paths'
    var_6 = 'verbose'
    var_7 = []
    var_8 = False
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = 'Please pass at least one file or directory'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_toml_path_is_file. Retrieved 10/20 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'Args'
    var_1 = 'config'
    var_2 = 'dummy.toml'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'verbose'
    var_6 = False
    var_7 = '--config'
    var_8 = [var_7, var_2]
    var_9 = module_0.make_config(var_8, var_4)



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'debug'
    var_2 = 'name'
    var_3 = 10
    var_4 = False
    var_5 = 'service'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 5
    var_8 = True
    var_9 = 'app'
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)

import vulture.config as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'unknown_key'
    var_4 = 1
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
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_config_with_existing_toml_file. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'verbose = true\n'
    var_1 = 'pyproject.toml'
    var_2 = 'vulture'
    var_3 = '--config'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_predicate_false_when_toml_path_is_not_a_file. Retrieved 6/10 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)



# Parsed testcases at query #23
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['min_confidence']
    assert var_5 == 50



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_predicate_false_via_non_existent_file. Retrieved 6/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)
    var_6 = bool(var_5 == {})
    assert var_6 is True



# Parsed testcases at query #25
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
    var_8 = "Data type for timeout must be 'int'"



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_evaluates_true_at_line_26. Retrieved 2/19 statements.


def test_case_0():
    var_0 = b'verbose = true\n'
    var_1 = '--config'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_input_config_raises_error_on_type_mismatch. Retrieved 8/11 statements.


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
    var_8 = "Data type for timeout must be 'int'"



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_config_toml_path_is_not_a_file. Retrieved 5/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = '--config'
    var_3 = [var_2, var_1]
    var_4 = module_0.make_config(var_3)



