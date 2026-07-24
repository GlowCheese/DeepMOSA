####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_config_with_toml_file_uses_toml_settings. Retrieved 2/6 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/8 statements.
# Partially parsed test_make_config_unknown_toml_key_raises_input_error. Retrieved 2/7 statements.
# Partially parsed test_make_config_wrong_type_in_toml_raises_input_error. Retrieved 2/7 statements.
# Partially parsed test_make_config_verbose_with_toml_path_prints_message. Retrieved 2/7 statements.
# Partially parsed test_make_config_config_not_overridden_by_toml. Retrieved 3/7 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == [])
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
    var_14 = var_2['config']
    assert var_14 == 'pyproject.toml'
    var_15 = var_2['verbose']
    assert var_15 is False

import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--min-confidence=80'
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = var_6['verbose']
    assert var_7 is True
    var_8 = var_6['min_confidence']
    assert var_8 == 80
    var_9 = var_6['paths']
    var_10 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_10 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_names = ["helper"]\nmin_confidence = 50\npaths = ["src/"]\n'
    var_1 = []

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    var_1 = '--min-confidence=80'
    var_2 = '--verbose=false'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key=value'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unknown configuration key'

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence=abc'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "Data type for min_confidence must be 'int'"

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Data type for min_confidence must be 'int'"

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Please pass at least one file or directory'

import vulture.config as module_0

def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['exclude']
    var_5 = bool(var_3['exclude'] == [])
    assert var_5 is True
    var_6 = var_3['ignore_decorators']
    var_7 = bool(var_3['ignore_decorators'] == [])
    assert var_7 is True
    var_8 = var_3['ignore_names']
    var_9 = bool(var_3['ignore_names'] == [])
    assert var_9 is True
    var_10 = var_3['make_whitelist']
    assert var_10 is False
    var_11 = var_3['min_confidence']
    assert var_11 == 60
    var_12 = var_3['sort_by_size']
    assert var_12 is False
    var_13 = var_3['verbose']
    assert var_13 is False
    var_14 = var_3['paths']
    var_15 = bool(var_3['paths'] == ['some_path'])
    assert var_15 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\nverbose = true\npaths = ["src/"]\n'
    var_1 = []
    var_2 = 'Reading configuration from'
    var_3 = 'src/'

import vulture.config as module_0

def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['config']
    assert var_4 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n[tool.vulture]\nconfig = "custom.toml"\npaths = ["src/"]\n'
    var_1 = '--config=custom.toml'
    var_2 = [var_1]



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'width'
    var_1 = 'height'
    var_2 = 'fullscreen'
    var_3 = 800
    var_4 = 600
    var_5 = False
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._check_input_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'fullscreen'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'width'
    var_1 = '800'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'height'
    var_1 = 600.0
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown1'
    var_1 = 'width'
    var_2 = 1
    var_3 = 'bad'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._check_input_config(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 21/23 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 21/23 statements.
# Partially parsed test_make_config_with_empty_toml_section. Retrieved 21/23 statements.
# Partially parsed test_make_config_with_missing_paths_raises_input_error. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_unknown_key_in_toml_raises_input_error. Retrieved 2/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'paths'
    var_5 = 'exclude'
    var_6 = 'ignore_decorators'
    var_7 = 'ignore_names'
    var_8 = 'make_whitelist'
    var_9 = 'min_confidence'
    var_10 = 'sort_by_size'
    var_11 = 'config'
    var_12 = 'verbose'
    var_13 = [var_1, var_2]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = 'pyproject.toml'
    var_19 = True
    var_20 = {var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_17, var_10: var_17, var_11: var_18, var_12: var_19}
    var_21 = None
    var_22 = module_0.make_config(var_3, var_21)
    var_23 = bool(var_22 == var_20)
    assert var_23 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\npaths = ["src"]'
    var_1 = None
    var_2 = 'paths'
    var_3 = 'exclude'
    var_4 = 'ignore_decorators'
    var_5 = 'ignore_names'
    var_6 = 'make_whitelist'
    var_7 = 'min_confidence'
    var_8 = 'sort_by_size'
    var_9 = 'config'
    var_10 = 'verbose'
    var_11 = 'src'
    var_12 = [var_11]
    var_13 = 'file*.py'
    var_14 = [var_13]
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = 'pyproject.toml'
    var_19 = True
    var_20 = {var_2: var_12, var_3: var_14, var_4: var_15, var_5: var_16, var_6: var_17, var_7: var_17, var_8: var_17, var_9: var_18, var_10: var_19}

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = false\npaths = ["toml_path"]'
    var_1 = '--verbose'
    var_2 = 'cli_path'
    var_3 = [var_1, var_2]
    var_4 = 'paths'
    var_5 = 'exclude'
    var_6 = 'ignore_decorators'
    var_7 = 'ignore_names'
    var_8 = 'make_whitelist'
    var_9 = 'min_confidence'
    var_10 = 'sort_by_size'
    var_11 = 'config'
    var_12 = 'verbose'
    var_13 = [var_2]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = 'pyproject.toml'
    var_19 = True
    var_20 = {var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_17, var_10: var_17, var_11: var_18, var_12: var_19}

def test_case_0():
    var_0 = b'[tool.vulture]'
    var_1 = '--sort-by-size'
    var_2 = 'some_path'
    var_3 = [var_1, var_2]
    var_4 = 'paths'
    var_5 = 'exclude'
    var_6 = 'ignore_decorators'
    var_7 = 'ignore_names'
    var_8 = 'make_whitelist'
    var_9 = 'min_confidence'
    var_10 = 'sort_by_size'
    var_11 = 'config'
    var_12 = 'verbose'
    var_13 = [var_2]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = True
    var_19 = 'pyproject.toml'
    var_20 = {var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_17, var_10: var_18, var_11: var_19, var_12: var_17}

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nunknown_key = true'
    var_1 = None
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unknown configuration key'



# Parsed testcases at query #4
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': ['path1', 'path2']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'pattern1,pattern2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'exclude': ['pattern1', 'pattern2']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = 'dec1,dec2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_decorators': ['dec1', 'dec2']})
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
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'config': 'custom.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'verbose': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = '--make-whitelist'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = bool(var_5 == {'verbose': True, 'make_whitelist': True, 'min_confidence': 80})
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 2/5 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/9 statements.
# Partially parsed test_make_config_with_empty_paths_raises_error. Retrieved 2/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.make_config(var_2, var_3)
    var_5 = var_4['paths']
    var_6 = bool(var_4['paths'] == ['path1', 'path2'])
    assert var_6 is True
    var_7 = var_4['verbose']
    assert var_7 is False
    var_8 = var_4['sort_by_size']
    assert var_8 is False
    var_9 = var_4['make_whitelist']
    assert var_9 is False
    var_10 = var_4['min_confidence']
    assert var_10 == 0
    var_11 = var_4['ignore_decorators']
    var_12 = bool(var_4['ignore_decorators'] == [])
    assert var_12 is True
    var_13 = var_4['ignore_names']
    var_14 = bool(var_4['ignore_names'] == [])
    assert var_14 is True
    var_15 = var_4['exclude']
    var_16 = bool(var_4['exclude'] == [])
    assert var_16 is True

def test_case_0():
    var_0 = b'\n[tool.vulture]\npaths = ["path1", "path2"]\nverbose = true\nsort_by_size = true\nmin_confidence = 50\n'
    var_1 = []

def test_case_0():
    var_0 = b'\n[tool.vulture]\npaths = ["toml_path"]\nverbose = false\nmin_confidence = 30\n'
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'cli_path'
    var_5 = [var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = b'\n[tool.vulture]\npaths = []\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'

import vulture.config as module_0

def test_case_0():
    var_0 = 'path'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['path'])
    assert var_5 is True
    var_6 = var_3['verbose']
    assert var_6 is False



# Parsed testcases at query #6
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 'hello'
    var_5 = 3.14
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 10
    var_8 = 'world'
    var_9 = 2.71
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_toml_path_not_file_returns_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = '/nonexistent/path/pyproject.toml'
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_dir_for_test'
    var_1 = 'config'
    var_2 = 'nonexistent.toml'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_with_only_cli_args. Retrieved 6/16 statements.
# Partially parsed test_make_config_with_toml_file. Retrieved 3/13 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/15 statements.
# Partially parsed test_make_config_defaults_applied. Retrieved 3/15 statements.
# Partially parsed test_make_config_empty_toml_uses_defaults. Retrieved 3/15 statements.
# Partially parsed test_make_config_without_toml_file_and_no_cli_paths. Retrieved 5/16 statements.
# Partially parsed test_make_config_verbose_with_toml_file. Retrieved 3/18 statements.
# Partially parsed test_make_config_no_verbose_with_toml_file. Retrieved 3/18 statements.
# Partially parsed test_make_config_unknown_key_in_toml. Retrieved 3/14 statements.
# Partially parsed test_make_config_wrong_type_in_toml. Retrieved 3/14 statements.
# Partially parsed test_make_config_no_paths_raises_error. Retrieved 3/14 statements.
# Partially parsed test_make_config_with_cli_paths_only. Retrieved 5/15 statements.
# Partially parsed test_make_config_with_toml_and_cli_paths. Retrieved 4/14 statements.
# Partially parsed test_make_config_verbose_from_toml. Retrieved 3/14 statements.
# Partially parsed test_make_config_verbose_false_from_toml. Retrieved 3/14 statements.
# Partially parsed test_make_config_min_confidence_from_toml. Retrieved 3/13 statements.
# Partially parsed test_make_config_sort_by_size_from_toml. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 0
    var_1 = '--verbose'
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_1, var_2, var_3]
    var_5 = ''

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = true\npaths = ["path1", "path2"]\n'
    var_2 = []

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = false\npaths = ["toml_path"]\n'
    var_2 = '--verbose'
    var_3 = 'cli_path'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = ''

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = ''

import vulture.config as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'pyproject.toml'
    var_2 = '--verbose'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = 'paths'
    var_6 = bool('paths' in var_4)
    assert var_6 is True

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = true\npaths = ["path1"]\n'
    var_2 = []
    var_3 = 'Reading configuration from'

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    var_2 = []

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nunknown_key = true\npaths = ["path1"]\n'
    var_2 = []
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Unknown configuration key: unknown_key'

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = "yes"\npaths = ["path1"]\n'
    var_2 = []
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Data type for verbose must be'

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = true\n'
    var_2 = []
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Please pass at least one file or directory'

def test_case_0():
    var_0 = 0
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_1, var_2]
    var_4 = ''

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\npaths = ["toml_path"]\n'
    var_2 = 'cli_path'
    var_3 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = true\npaths = ["path1"]\n'
    var_2 = []

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    var_2 = []

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nmin_confidence = 50\npaths = ["path1"]\n'
    var_2 = []

def test_case_0():
    var_0 = 0
    var_1 = '\n[tool.vulture]\nsort_by_size = true\npaths = ["path1"]\n'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_true_when_toml_path_is_file. Retrieved 8/25 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = b'[tool.vulture]\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False
    var_4 = 'test'
    var_5 = [var_4]
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)



# Parsed testcases at query #11
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'wrong_type'
    var_2 = {var_0: var_1}
    var_3 = 42
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_with_toml_file_override. Retrieved 5/14 statements.
# Partially parsed test_make_config_with_defaults. Retrieved 2/11 statements.
# Partially parsed test_make_config_with_auto_detected_toml. Retrieved 4/14 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 3/12 statements.
# Partially parsed test_make_config_raises_on_unknown_key. Retrieved 2/12 statements.
# Partially parsed test_make_config_raises_on_empty_paths. Retrieved 2/12 statements.
# Partially parsed test_make_config_with_no_toml_and_no_cli_paths. Retrieved 3/14 statements.
# Partially parsed test_make_config_with_toml_file_and_no_cli_paths. Retrieved 2/12 statements.
# Partially parsed test_make_config_verbose_with_toml_path. Retrieved 2/12 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--sort-by-size'
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = var_5['verbose']
    assert var_6 is True
    var_7 = var_5['sort_by_size']
    assert var_7 is True
    var_8 = var_5['paths']
    var_9 = bool(var_5['paths'] == ['path1', 'path2'])
    assert var_9 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = false\nsort_by_size = false\npaths = ["toml_path"]\n'
    var_1 = '--verbose'
    var_2 = '--sort-by-size'
    var_3 = 'cli_path'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = ["test_path"]\n'
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.vulture]\nverbose = true\npaths = ["auto_path"]\n'
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['verbose']
    assert var_4 is True
    var_5 = var_3['paths']
    var_6 = bool(var_3['paths'] == ['auto_path'])
    assert var_6 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = false\npaths = ["toml_path"]\n'
    var_1 = '--verbose'
    var_2 = [var_1]

def test_case_0():
    var_0 = b'[tool.vulture]\nunknown_key = true\npaths = ["test"]\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unknown configuration key'

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = []\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'

import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == [])
    assert var_4 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nsort_by_size = true\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\npaths = ["test_path"]\n'
    var_1 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_config_with_only_cli_args. Retrieved 5/13 statements.
# Partially parsed test_make_config_with_toml_and_cli. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_defaults. Retrieved 2/5 statements.


def test_case_0():
    var_0 = b'[tool.vulture]\n'
    var_1 = '--verbose'
    var_2 = 'path1'
    var_3 = [var_1, var_2]
    var_4 = b''

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    var_1 = '--verbose'
    var_2 = 'path2'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = ["path1"]\n'
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = 'path1'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['verbose']
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'string'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'known_key'
    var_4 = 'default'
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '/nonexistent/path/that/does/not/exist'



# Parsed testcases at query #16
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_toml_path_is_file_and_reads_config. Retrieved 2/11 statements.


def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = []\n'
    var_1 = '--config'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_toml_path_not_file. Retrieved 7/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = None
    var_1 = '--config'
    var_2 = '/nonexistent/path/to/config'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3, var_0)
    var_5 = 'verbose'
    var_6 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_true_when_toml_path_is_file. Retrieved 3/13 statements.


def test_case_0():
    var_0 = False
    var_1 = '--config'
    var_2 = None



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_toml_path_is_file_opens_and_parses_toml. Retrieved 2/16 statements.


def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["test"]\n'
    var_1 = 'exclude'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'nonexistent.toml'
    var_1 = 'config'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_line26_evaluates_to_false. Retrieved 2/14 statements.


def test_case_0():
    var_0 = '.toml'
    var_1 = '--config'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_config_predicate_line26_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = b''
    var_1 = '--config'



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'batch_size'
    var_1 = 'learning_rate'
    var_2 = 'epochs'
    var_3 = 32
    var_4 = 0.01
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 64
    var_8 = 0.001
    var_9 = 5
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'batch_size'
    var_4 = 64
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'batch_size'
    var_1 = '32'
    var_2 = {var_0: var_1}
    var_3 = 64
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_2)
    var_5 = bool(False)
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'batch_size'
    var_2 = 64
    var_3 = {var_1: var_2}
    var_4 = module_0._check_input_config(var_0)



# Parsed testcases at query #26
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    assert var_5 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_make_config_with_cli_args_overriding_toml. Retrieved 5/7 statements.
# Partially parsed test_make_config_with_toml_only. Retrieved 2/4 statements.
# Partially parsed test_make_config_with_both_toml_and_cli_defaults. Retrieved 3/5 statements.
# Partially parsed test_make_config_with_toml_unknown_key. Retrieved 2/6 statements.
# Partially parsed test_make_config_with_toml_wrong_type. Retrieved 2/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == [])
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
    var_12 = var_2['sort_by_size']
    assert var_12 is False
    var_13 = var_2['verbose']
    assert var_13 is False

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = ["default_path"]\nverbose = true\n'
    var_1 = '--verbose'
    var_2 = '--sort-by-size'
    var_3 = 'cli_path'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["*.pyc"]\nmake_whitelist = true\n'
    var_1 = []

def test_case_0():
    var_0 = b'[tool.vulture]\nignore_names = ["test_*"]\n'
    var_1 = '--verbose'
    var_2 = [var_1]

def test_case_0():
    var_0 = b'[tool.vulture]\nunknown_key = true\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unknown configuration key'

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = 123\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Data type for verbose must be'

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Please pass at least one file or directory'

import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_toml_path_is_file_evaluates_to_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = False
    var_1 = '--config'



# Parsed testcases at query #29
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'value1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_2)
    var_5 = bool(False)
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 5/11 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'nonexistent_config.toml'
    var_1 = 'config'
    var_2 = {var_1: var_0}
    var_3 = module_0._parse_args(var_2)
    var_4 = var_3[var_1]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = []'
    var_1 = '--config'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0._check_output_config(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': ['path1', 'path2']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = '*.py,docs'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'exclude': ['*.py', 'docs']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@app.route,@require_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_decorators': ['@app.route', '@require_*']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'visit_*,do_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'ignore_names': ['visit_*', 'do_*']})
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
    var_1 = '80'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'min_confidence': 80})
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
    var_1 = 'custom.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'config': 'custom.toml'})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = [var_0]
    var_2 = module_0._parse_args(var_1)
    var_3 = bool(var_2 == {'verbose': True})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '-v'
    var_1 = '--sort-by-size'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = bool(var_5 == {'verbose': True, 'sort_by_size': True, 'min_confidence': 50})
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 2/5 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_empty_toml_and_no_cli. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_no_toml_and_no_cli_paths_raises_error. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_toml_path_argument. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_defaults_for_missing_options. Retrieved 2/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--make-whitelist'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['make_whitelist']
    assert var_5 is True
    var_6 = var_4['paths']
    var_7 = bool(var_4['paths'] == ['path1', 'path2'])
    assert var_7 is True
    var_8 = var_4['verbose']
    assert var_8 is False
    var_9 = var_4['exclude']
    var_10 = bool(var_4['exclude'] == [])
    assert var_10 is True
    var_11 = var_4['ignore_decorators']
    var_12 = bool(var_4['ignore_decorators'] == [])
    assert var_12 is True
    var_13 = var_4['ignore_names']
    var_14 = bool(var_4['ignore_names'] == [])
    assert var_14 is True
    var_15 = var_4['min_confidence']
    assert var_15 == 0
    var_16 = var_4['sort_by_size']
    assert var_16 is False
    var_17 = var_4['config']
    assert var_17 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_1 = []

def test_case_0():
    var_0 = '\n[tool.vulture]\nmake_whitelist = false\nverbose = false\npaths = ["toml_path"]\n'
    var_1 = '--make-whitelist'
    var_2 = '--verbose'
    var_3 = 'cli_path'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = ''
    var_1 = []

def test_case_0():
    var_0 = ''
    var_1 = []

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["path_from_toml"]\n'
    var_1 = '--config'
    var_2 = 'some_config'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["test_path"]\n'
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = 'path3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['paths']
    var_6 = bool(var_4['paths'] == ['path1', 'path2', 'path3'])
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = '*.py,test*'
    var_2 = 'path'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['exclude']
    var_6 = bool(var_4['exclude'] == ['*.py', 'test*'])
    assert var_6 is True
    var_7 = var_4['paths']
    var_8 = bool(var_4['paths'] == ['path'])
    assert var_8 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = 'path'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['min_confidence']
    assert var_5 == 50

import vulture.config as module_0

def test_case_0():
    var_0 = '--sort-by-size'
    var_1 = 'path'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['sort_by_size']
    assert var_4 is True
    var_5 = var_3['paths']
    var_6 = bool(var_3['paths'] == ['path'])
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = 'non_existent_file.toml'
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]



# Parsed testcases at query #5
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = 'port'
    var_2 = 'name'
    var_3 = True
    var_4 = 8080
    var_5 = 'test'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._check_input_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = 'true'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = '8080'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._check_input_config(var_0)



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = 'verbose'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = bool(var_3 and var_2['verbose'])
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 'verbose'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '/some/path/pyproject.toml'
    var_4 = bool(var_3 and var_2['verbose'])
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_config_with_no_argv_and_no_tomlfile_uses_defaults. Retrieved 2/9 statements.
# Partially parsed test_make_config_with_cli_arguments_overrides_toml. Retrieved 5/14 statements.
# Partially parsed test_make_config_with_tomlfile_sets_config_and_detected_path. Retrieved 2/11 statements.
# Partially parsed test_make_config_with_empty_argv_and_existing_toml_file. Retrieved 5/18 statements.
# Partially parsed test_make_config_with_missing_toml_file_uses_defaults. Retrieved 4/15 statements.
# Partially parsed test_make_config_raises_input_error_when_paths_empty. Retrieved 2/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = module_0.make_config()
    var_2 = var_1['paths']
    var_3 = bool(var_1['paths'] == [])
    assert var_3 is True
    var_4 = var_1['exclude']
    assert var_4 is None
    var_5 = var_1['ignore_decorators']
    assert var_5 is None
    var_6 = var_1['ignore_names']
    assert var_6 is None
    var_7 = var_1['make_whitelist']
    assert var_7 is False
    var_8 = var_1['min_confidence']
    assert var_8 is None
    var_9 = var_1['sort_by_size']
    assert var_9 is False
    var_10 = var_1['verbose']
    assert var_10 is False
    var_11 = var_1['config']
    assert var_11 == 'pyproject.toml'

def test_case_0():
    var_0 = b'\n[tool.vulture]\nexclude = ["file*.py"]\nverbose = false\n'
    var_1 = 'vulture'
    var_2 = '--verbose'
    var_3 = '--exclude'
    var_4 = 'other*.py'

def test_case_0():
    var_0 = b'\n[tool.vulture]\npaths = ["src"]\nverbose = true\n'
    var_1 = 'vulture'

import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = b'\n[tool.vulture]\nsort_by_size = true\n'
    var_2 = 'vulture'
    var_3 = '--config'
    var_4 = module_0.make_config()
    var_5 = var_4['sort_by_size']
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'nonexistent.toml'
    var_1 = 'vulture'
    var_2 = '--config'
    var_3 = module_0.make_config()
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == [])
    assert var_5 is True
    var_6 = var_3['sort_by_size']
    assert var_6 is False

import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = module_0.make_config()
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 10
    var_3 = 'hello'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = 'world'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_25_returns_false_when_toml_path_is_not_file. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = '/nonexistent/path/pyproject.toml'
    var_2 = {var_0: var_1}
    var_3 = 'config'
    var_4 = var_2[var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_toml_path_is_not_file. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = '/nonexistent/path/pyproject.toml'
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_overrides. Retrieved 5/9 statements.
# Partially parsed test_make_config_with_toml_and_defaults. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_no_paths_raises_error. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["path1"]\nmin_confidence = 60\n'
    var_1 = '--sort-by-size'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['paths']
    var_6 = bool(var_4['paths'] == ['path1', 'path2'])
    assert var_6 is True
    var_7 = var_4['verbose']
    assert var_7 is True
    var_8 = var_4['sort_by_size']
    assert var_8 is False

def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["path1"]\n'
    var_1 = []

def test_case_0():
    var_0 = '[tool.vulture]\npaths = []\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = 'host'
    var_2 = 'debug'
    var_3 = 8080
    var_4 = 'localhost'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0._check_input_config(var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'port'
    var_1 = '8080'
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'host'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = bool(False)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._check_input_config(var_0)



# Parsed testcases at query #14
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #15
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = 100
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 2/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/7 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == [])
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
    var_14 = var_2['config']
    assert var_14 == 'pyproject.toml'
    var_15 = var_2['verbose']
    assert var_15 is False

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test.py,utils'
    var_2 = '--verbose'
    var_3 = 'src'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = var_6['exclude']
    var_8 = bool(var_6['exclude'] == ['test.py', 'utils'])
    assert var_8 is True
    var_9 = var_6['verbose']
    assert var_9 is True
    var_10 = var_6['paths']
    var_11 = bool(var_6['paths'] == ['src'])
    assert var_11 is True

def test_case_0():
    var_0 = "[tool.vulture]\nexclude = ['file1.py', 'dir/']\nverbose = true\npaths = ['path1']"
    var_1 = []

def test_case_0():
    var_0 = "[tool.vulture]\nverbose = true\npaths = ['toml_path']"
    var_1 = '--verbose'
    var_2 = False
    var_3 = 'cli_path'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_line26_evaluates_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = b"key = 'value'\n"
    var_1 = '--config'
    var_2 = 'config'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_with_no_args_and_no_tomlfile. Retrieved 8/29 statements.
# Partially parsed test_make_config_with_paths_and_no_tomlfile. Retrieved 12/28 statements.
# Partially parsed test_make_config_with_tomlfile. Retrieved 3/15 statements.
# Partially parsed test_make_config_with_cli_overrides_toml. Retrieved 7/19 statements.
# Partially parsed test_make_config_with_paths_and_tomlfile. Retrieved 4/16 statements.
# Partially parsed test_make_config_with_tomlfile_no_section. Retrieved 3/15 statements.
# Partially parsed test_make_config_with_empty_toml. Retrieved 4/14 statements.
# Partially parsed test_make_config_with_verbose_and_tomlfile. Retrieved 3/23 statements.
# Partially parsed test_make_config_with_default_toml_file. Retrieved 11/37 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = []
    var_2 = module_0._parse_args(var_1)
    var_3 = 'config'
    var_4 = var_2[var_3]
    var_5 = {}
    var_6 = print(var_1)
    var_7 = module_0._check_output_config(var_5)
    var_8 = var_5['paths']
    var_9 = bool(var_5['paths'] == [])
    assert var_9 is True
    var_10 = var_5['exclude']
    assert var_10 is None
    var_11 = var_5['ignore_decorators']
    assert var_11 is None
    var_12 = var_5['ignore_names']
    assert var_12 is None
    var_13 = var_5['make_whitelist']
    assert var_13 is False
    var_14 = var_5['min_confidence']
    assert var_14 is None
    var_15 = var_5['sort_by_size']
    assert var_15 is False
    var_16 = var_5['config']
    assert var_16 == 'pyproject.toml'
    var_17 = var_5['verbose']
    assert var_17 is False

import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_3, var_4]
    var_6 = module_0._parse_args(var_5)
    var_7 = 'config'
    var_8 = var_6[var_7]
    var_9 = {}
    var_10 = print(var_3)
    var_11 = module_0._check_output_config(var_9)
    var_12 = var_9['paths']
    var_13 = bool(var_9['paths'] == ['path1', 'path2'])
    assert var_13 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1"]\n'
    var_1 = []
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nignore_names = ["name1"]\nmake_whitelist = false\nmin_confidence = 10\nsort_by_size = true\nverbose = false\npaths = ["path1"]\n'
    var_1 = '--exclude'
    var_2 = 'other.py'
    var_3 = '--verbose'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0._parse_args(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["toml_path"]\n'
    var_1 = 'cli_path'
    var_2 = [var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '[other]\nkey = "value"\n'
    var_1 = []
    var_2 = module_0._parse_args(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = b''
    var_1 = []
    var_2 = module_0._parse_args(var_1)
    var_3 = print(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true\n'
    var_1 = []
    var_2 = module_0._parse_args(var_1)
    var_3 = 'Reading configuration from'
    var_4 = bool('Reading configuration from' in var_1)
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = 'pyproject.toml'
    var_2 = var_0 / var_1
    var_3 = '[tool.vulture]\nverbose = true\n'
    var_4 = []
    var_5 = module_0._parse_args(var_4)
    var_6 = 'config'
    var_7 = var_5[var_6]
    var_8 = {}
    var_9 = print(var_0)
    var_10 = module_0._check_output_config(var_8)
    var_11 = var_8['verbose']
    assert var_11 is True

def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_line_26_true. Retrieved 5/21 statements.


def test_case_0():
    var_0 = False
    var_1 = '.toml'
    var_2 = 'config'
    var_3 = None
    var_4 = {}



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_config_with_cli_args_and_tomlfile. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_tomlfile_overwrites_defaults. Retrieved 3/6 statements.
# Partially parsed test_make_config_with_cli_overrides_toml. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_unknown_key_in_toml_raises_error. Retrieved 3/7 statements.
# Partially parsed test_make_config_with_tomlfile_and_no_cli_paths. Retrieved 2/5 statements.


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

def test_case_0():
    var_0 = '[tool.vulture]\nexclude = ["file*.py"]\n'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 80\n'
    var_1 = 'path3'
    var_2 = [var_1]

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = false\n'
    var_1 = 'path4'
    var_2 = '--verbose'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = '[tool.vulture]\nunknown_key = true\n'
    var_1 = 'path5'
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["path6"]\n'
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'path7'
    var_1 = '--exclude'
    var_2 = '*.pyc'
    var_3 = '--ignore-decorators'
    var_4 = 'deco1'
    var_5 = '--ignore-names'
    var_6 = 'name1'
    var_7 = '--make-whitelist'
    var_8 = '--min-confidence'
    var_9 = '90'
    var_10 = '--sort-by-size'
    var_11 = '--verbose'
    var_12 = '--config'
    var_13 = 'custom.toml'
    var_14 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = var_15['exclude']
    var_17 = bool(var_15['exclude'] == ['*.pyc'])
    assert var_17 is True
    var_18 = var_15['ignore_decorators']
    var_19 = bool(var_15['ignore_decorators'] == ['deco1'])
    assert var_19 is True
    var_20 = var_15['ignore_names']
    var_21 = bool(var_15['ignore_names'] == ['name1'])
    assert var_21 is True
    var_22 = var_15['make_whitelist']
    assert var_22 is True
    var_23 = var_15['min_confidence']
    assert var_23 == 90
    var_24 = var_15['sort_by_size']
    assert var_24 is True
    var_25 = var_15['verbose']
    assert var_25 is True
    var_26 = var_15['config']
    assert var_26 == 'custom.toml'



# Parsed testcases at query #22
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude=test.py'
    var_1 = 'src'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.make_config(var_2, var_3)
    var_5 = var_4['exclude']
    var_6 = bool(var_4['exclude'] == ['test.py'])
    assert var_6 is True
    var_7 = var_4['paths']
    var_8 = bool(var_4['paths'] == ['src'])
    assert var_8 is True
    var_9 = var_4['make_whitelist']
    assert var_9 is False
    var_10 = var_4['min_confidence']
    assert var_10 == 80



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_config_with_tomlfile. Retrieved 1/5 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 3/7 statements.
# Partially parsed test_make_config_defaults_applied. Retrieved 3/7 statements.
# Partially parsed test_make_config_raises_on_empty_paths. Retrieved 2/7 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
    assert var_4 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["dir1", "dir2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["toml_path"]\n'
    var_1 = 'cli_path.py'
    var_2 = [var_1]

def test_case_0():
    var_0 = '\n[tool.vulture]\nverbose = true\n'
    var_1 = 'file.py'
    var_2 = [var_1]

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = []\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = '/nonexistent/path/pyproject.toml'
    var_1 = 'config'
    var_2 = {var_1: var_0}
    var_3 = None



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #26
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = bool(var_2 == {'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'min_confidence': 0, 'sort_by_size': False, 'config': 'pyproject.toml', 'verbose': False})
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_line_26_true_when_toml_path_is_file. Retrieved 4/20 statements.


def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = []\n'
    var_1 = None
    var_2 = '--config'
    var_3 = 'config'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_config_with_no_arguments_uses_defaults. Retrieved 1/9 statements.
# Partially parsed test_make_config_cli_args_override_toml. Retrieved 4/7 statements.
# Partially parsed test_make_config_toml_provided_without_cli. Retrieved 1/4 statements.
# Partially parsed test_make_config_raises_input_error_for_empty_paths. Retrieved 1/5 statements.
# Partially parsed test_make_config_sets_defaults_for_missing_toml_keys. Retrieved 1/4 statements.
# Partially parsed test_make_config_detects_toml_file_in_current_directory. Retrieved 3/14 statements.
# Partially parsed test_make_config_uses_config_argument_for_toml_path. Retrieved 3/17 statements.
# Partially parsed test_make_config_with_verbose_and_toml_file. Retrieved 1/4 statements.
# Failed to parse test_make_config_with_no_toml_and_no_cli_args.


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = var_0['paths']
    var_2 = bool(var_0['paths'] == [])
    assert var_2 is True
    var_3 = var_0['min_confidence']
    assert var_3 == 0
    var_4 = var_0['sort_by_size']
    assert var_4 is False
    var_5 = var_0['verbose']
    assert var_5 is False
    var_6 = var_0['make_whitelist']
    assert var_6 is False
    var_7 = var_0['exclude']
    var_8 = bool(var_0['exclude'] == [])
    assert var_8 is True
    var_9 = var_0['ignore_decorators']
    var_10 = bool(var_0['ignore_decorators'] == [])
    assert var_10 is True
    var_11 = var_0['ignore_names']
    var_12 = bool(var_0['ignore_names'] == [])
    assert var_12 is True
    var_13 = var_0['config']
    assert var_13 == 'pyproject.toml'

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\nverbose = true\n'
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["src"]\n'

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = []\n'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = ["test"]\n'

import vulture.config as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = '--help'
    var_5 = [var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = bool(False)
    assert var_7 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.vulture]\nmin_confidence = 40\n'
    var_2 = module_0.make_config()
    var_3 = var_2['min_confidence']
    assert var_3 == 40

def test_case_0():
    var_0 = 'custom_config.toml'
    var_1 = '[tool.vulture]\nmin_confidence = 60\n'
    var_2 = '--config'

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\npaths = ["src"]\n'



# Parsed testcases at query #29
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'param1'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_true_when_toml_path_is_file. Retrieved 4/22 statements.


def test_case_0():
    var_0 = False
    var_1 = '.toml'
    var_2 = b'[tool.vulture]'
    var_3 = 'config'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_config_with_tomlfile. Retrieved 2/6 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 3/6 statements.
# Partially parsed test_make_config_defaults_for_missing_options. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_invalid_key_in_toml. Retrieved 2/7 statements.
# Partially parsed test_make_config_no_paths_raises_error. Retrieved 2/6 statements.
# Partially parsed test_make_config_with_toml_path_from_cli. Retrieved 3/13 statements.
# Partially parsed test_make_config_verbose_prints_detected_path. Retrieved 2/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_8 is True
    var_9 = var_6['verbose']
    assert var_9 is True
    var_10 = var_6['sort_by_size']
    assert var_10 is True

def test_case_0():
    var_0 = b'[tool.vulture]\nexclude = ["file*.py"]\nverbose = true\npaths = ["path1", "path2"]\n'
    var_1 = []

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = false\npaths = ["path1"]\n'
    var_1 = '--verbose'
    var_2 = [var_1]

def test_case_0():
    var_0 = b'[tool.vulture]\n'
    var_1 = []

def test_case_0():
    var_0 = b'[tool.vulture]\nunknown_key = "value"\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Unknown configuration key'

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-flag'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = []\n'
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Please pass at least one file or directory'

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\npaths = ["path1"]\n'
    var_1 = '--config'
    var_2 = 'path2'
    var_3 = 'path2'
    var_4 = 'path1'

def test_case_0():
    var_0 = b'[tool.vulture]\nverbose = true\n'
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = '--config'
    var_2 = '/nonexistent/path.toml'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['paths']
    var_6 = bool(var_4['paths'] == ['path1'])
    assert var_6 is True
    var_7 = var_4['verbose']
    assert var_7 is False



# Parsed testcases at query #32
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'max_iterations'
    var_1 = 'tolerance'
    var_2 = 100
    var_3 = 1e-05
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0._check_input_config(var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/6 statements.
# Partially parsed test_make_config_toml_invalid_key_raises. Retrieved 1/4 statements.
# Partially parsed test_make_config_toml_without_vulture_section. Retrieved 1/3 statements.
# Partially parsed test_make_config_verbose_output. Retrieved 1/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = var_3['paths']
    var_5 = bool(var_3['paths'] == ['path1', 'path2'])
    assert var_5 is True

def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["dir1", "dir2"]\nverbose = true'

def test_case_0():
    var_0 = '[tool.vulture]\npaths = ["dir1"]\nverbose = false'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['paths']
    var_3 = bool(var_1['paths'] == [])
    assert var_3 is True
    var_4 = var_1['exclude']
    assert var_4 is None
    var_5 = var_1['ignore_decorators']
    assert var_5 is None
    var_6 = var_1['ignore_names']
    assert var_6 is None
    var_7 = var_1['make_whitelist']
    assert var_7 is False
    var_8 = var_1['min_confidence']
    assert var_8 is None
    var_9 = var_1['sort_by_size']
    assert var_9 is False
    var_10 = var_1['verbose']
    assert var_10 is False

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
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

def test_case_0():
    var_0 = '[tool.vulture]\ninvalid_key = true'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '[tool.other]\nkey = true'

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true'
    var_1 = 'Reading configuration from'

import vulture.config as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(False)
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = None
    var_1 = '--config'
    var_2 = 'nonexistent.toml'
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_toml_file_exists_and_is_file. Retrieved 5/16 statements.


def test_case_0():
    var_0 = False
    var_1 = '.toml'
    var_2 = b''
    var_3 = '--config'
    var_4 = None



# Parsed testcases at query #36
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)



