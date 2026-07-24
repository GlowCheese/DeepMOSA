####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_unknown_key_in_toml_raises_error. Retrieved 1/5 statements.
# Partially parsed test_make_config_wrong_type_in_toml_raises_error. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
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

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'file.py'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['file.py'])
    assert var_8 is True
    var_9 = var_6['exclude']
    var_10 = bool(var_6['exclude'] == ['test_*.py'])
    assert var_10 is True
    var_11 = var_6['min_confidence']
    assert var_11 == 80

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["path1", "path2"]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["path1"]\nmin_confidence = 10\n'
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = 'path2'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n[tool.vulture]\nunknown_key = "value"\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = 'test.toml'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_verbose_output_when_toml_detected. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = 'rb'
    var_4 = 'Reading configuration from'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--verbose'
    var_3 = 'src/'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = var_5['exclude']
    var_7 = bool(var_5['exclude'] == ['test_*.py'])
    assert var_7 is True
    var_8 = var_5['verbose']
    assert var_8 is True
    var_9 = var_5['paths']
    var_10 = bool(var_5['paths'] == ['src/'])
    assert var_10 is True

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    verbose = true\n    paths = ["src/"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    verbose = false\n    paths = ["src/"]\n    '
    var_1 = '--exclude'
    var_2 = 'docs/'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

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
    var_0 = '\n    [tool.vulture]\n    min_confidence = 90\n    verbose = true\n    paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 90\n    verbose = false\n    '
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(str(e).startswith('Unknown configuration key:'))
    assert var_5 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(str(e).startswith('Data type for min_confidence must be'))
    assert var_5 is True



# Parsed testcases at query #6
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
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_false. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = 'verbose'
    var_5 = 'detected_toml_path'



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

# Partially parsed test_predicate_at_line_39_evaluates_to_false. Retrieved 7/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'nonexistent.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'detected_toml_path'
    var_5 = 'verbose'
    var_6 = var_3[var_5]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #12
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 42
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'non_existent_file.toml'



# Parsed testcases at query #14
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default_value'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 'default_value'
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #15
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
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 0
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/7 statements.


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

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nverbose = true\npaths = ["path1", "path2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nverbose = false\n'
    var_1 = '--exclude'
    var_2 = 'other_*.py'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = var_0['exclude']
    var_2 = var_0['verbose']
    var_3 = var_0['paths']
    var_4 = bool(var_0['paths'] == [])
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()

import vulture.config as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/7 statements.


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
    var_11 = var_6['exclude']
    var_12 = bool(var_6['exclude'] == [])
    assert var_12 is True
    var_13 = var_6['ignore_decorators']
    var_14 = bool(var_6['ignore_decorators'] == [])
    assert var_14 is True
    var_15 = var_6['ignore_names']
    var_16 = bool(var_6['ignore_names'] == [])
    assert var_16 is True
    var_17 = var_6['make_whitelist']
    assert var_17 is False
    var_18 = var_6['sort_by_size']
    assert var_18 is False

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\n'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['exclude']
    var_3 = bool(var_1['exclude'] == [])
    assert var_3 is True
    var_4 = var_1['ignore_decorators']
    var_5 = bool(var_1['ignore_decorators'] == [])
    assert var_5 is True
    var_6 = var_1['ignore_names']
    var_7 = bool(var_1['ignore_names'] == [])
    assert var_7 is True
    var_8 = var_1['make_whitelist']
    assert var_8 is False
    var_9 = var_1['min_confidence']
    assert var_9 == 60
    var_10 = var_1['sort_by_size']
    assert var_10 is False
    var_11 = var_1['verbose']
    assert var_11 is False
    var_12 = var_1['paths']
    var_13 = bool(var_1['paths'] == [])
    assert var_13 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_toml_overrides. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 11/13 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['path1.py'])
    assert var_4 is True
    var_5 = var_2['exclude']
    var_6 = var_2['ignore_decorators']
    var_7 = var_2['ignore_names']
    var_8 = var_2['make_whitelist']
    var_9 = var_2['min_confidence']
    var_10 = var_2['sort_by_size']
    var_11 = var_2['verbose']

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*,*.pyc'
    var_2 = '--ignore-decorators'
    var_3 = 'deco1,deco2'
    var_4 = '--ignore-names'
    var_5 = 'name1,name2'
    var_6 = '--make-whitelist'
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = '--sort-by-size'
    var_10 = '--verbose'
    var_11 = 'path1.py'
    var_12 = 'path2.py'
    var_13 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = var_14['paths']
    var_16 = bool(var_14['paths'] == ['path1.py', 'path2.py'])
    assert var_16 is True
    var_17 = var_14['exclude']
    var_18 = bool(var_14['exclude'] == ['test_*', '*.pyc'])
    assert var_18 is True
    var_19 = var_14['ignore_decorators']
    var_20 = bool(var_14['ignore_decorators'] == ['deco1', 'deco2'])
    assert var_20 is True
    var_21 = var_14['ignore_names']
    var_22 = bool(var_14['ignore_names'] == ['name1', 'name2'])
    assert var_22 is True
    var_23 = var_14['make_whitelist']
    assert var_23 is True
    var_24 = var_14['min_confidence']
    assert var_24 == 50
    var_25 = var_14['sort_by_size']
    assert var_25 is True
    var_26 = var_14['verbose']
    assert var_26 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*", "*.pyc"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 50\nsort_by_size = true\nverbose = true\npaths = ["path1.py", "path2.py"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*", "*.pyc"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 50\nsort_by_size = true\nverbose = true\npaths = ["path1.py", "path2.py"]\n'
    var_1 = '--exclude'
    var_2 = 'new_test_*,*.pyc'
    var_3 = '--ignore-decorators'
    var_4 = 'new_deco1,new_deco2'
    var_5 = '--ignore-names'
    var_6 = 'new_name1,new_name2'
    var_7 = '--min-confidence'
    var_8 = '75'
    var_9 = 'path3.py'
    var_10 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key'
    var_1 = 'value'
    var_2 = 'path1.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = 'path1.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/non/existent/file.toml'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 4/14 statements.


def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 0.5\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False



# Parsed testcases at query #24
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent_file.toml'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_make_config_with_toml_file_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_both_cli_and_toml. Retrieved 7/10 statements.
# Partially parsed test_make_config_with_unknown_toml_key_raises_error. Retrieved 1/6 statements.
# Partially parsed test_make_config_with_wrong_type_toml_value_raises_error. Retrieved 1/6 statements.


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
    var_17 = var_4['config']
    assert var_17 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["toml_path1", "toml_path2"]\nexclude = ["*test*.py"]\nignore_decorators = ["@decorator1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 80\nsort_by_size = true\nverbose = true\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["toml_path1"]\nexclude = ["*test*.py"]\nignore_decorators = ["@decorator1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 80\nsort_by_size = true\nverbose = false\n'
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '90'
    var_4 = 'cli_path1'
    var_5 = 'cli_path2'
    var_6 = [var_1, var_2, var_3, var_4, var_5]

import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = '\n[tool.vulture]\nunknown_key = "value"\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_make_config_with_toml_file_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_cli_args_overriding_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_unknown_key_in_toml_raises_error. Retrieved 1/6 statements.
# Partially parsed test_make_config_with_wrong_type_in_toml_raises_error. Retrieved 1/6 statements.


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
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = 'path3'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = "not_an_integer"\n    '



# Parsed testcases at query #31
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_2)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_toml_path_is_file_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = '/some/path'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0._check_output_config(var_3)



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
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'paths': ['file1.py', 'file2.py']})
    assert var_4 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = '*.py,test_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)
    var_4 = bool(var_3 == {'exclude': ['*.py', 'test_*']})
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
    var_0 = 'file.py'
    var_1 = '--exclude'
    var_2 = '*.py'
    var_3 = '--verbose'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0._parse_args(var_4)
    var_6 = bool(var_5 == {'paths': ['file.py'], 'exclude': ['*.py'], 'verbose': True})
    assert var_6 is True

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_a_number'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/7 statements.


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

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        verbose = false\n        paths = ["toml_path"]\n    '
    var_1 = '--verbose'
    var_2 = 'cli_path'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #5
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default'
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
    var_5 = 'default'
    var_6 = {var_0: var_5}
    var_7 = module_0._check_input_config(var_4)
    var_8 = bool(False)
    assert var_8 is True

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


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
    var_11 = 'exclude'
    var_12 = bool('exclude' in var_6)
    assert var_12 is True
    var_13 = 'ignore_decorators'
    var_14 = bool('ignore_decorators' in var_6)
    assert var_14 is True
    var_15 = 'ignore_names'
    var_16 = bool('ignore_names' in var_6)
    assert var_16 is True
    var_17 = 'make_whitelist'
    var_18 = bool('make_whitelist' in var_6)
    assert var_18 is True
    var_19 = 'sort_by_size'
    var_20 = bool('sort_by_size' in var_6)
    assert var_20 is True

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = 10\n        verbose = false\n    '
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_config_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_invalid_toml_type. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

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
    var_11 = var_6['exclude']
    var_12 = bool(var_6['exclude'] == [])
    assert var_12 is True
    var_13 = var_6['ignore_decorators']
    var_14 = bool(var_6['ignore_decorators'] == [])
    assert var_14 is True
    var_15 = var_6['ignore_names']
    var_16 = bool(var_6['ignore_names'] == [])
    assert var_16 is True
    var_17 = var_6['make_whitelist']
    assert var_17 is False
    var_18 = var_6['sort_by_size']
    assert var_18 is False

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@deco1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 30\nsort_by_size = true\nverbose = true\npaths = ["path1"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nmin_confidence = 30\npaths = ["path1"]\n'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path2'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'not_an_int'
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_toml_path_is_file_predicate_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent_file.toml'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_toml_overrides. Retrieved 3/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
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
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--ignore-decorators'
    var_3 = 'deco1,deco2'
    var_4 = '--ignore-names'
    var_5 = 'name1,name2'
    var_6 = '--make-whitelist'
    var_7 = '--min-confidence'
    var_8 = '80'
    var_9 = '--sort-by-size'
    var_10 = '--verbose'
    var_11 = 'file.py'
    var_12 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = var_13['paths']
    var_15 = bool(var_13['paths'] == ['file.py'])
    assert var_15 is True
    var_16 = var_13['exclude']
    var_17 = bool(var_13['exclude'] == ['test_*.py'])
    assert var_17 is True
    var_18 = var_13['ignore_decorators']
    var_19 = bool(var_13['ignore_decorators'] == ['deco1', 'deco2'])
    assert var_19 is True
    var_20 = var_13['ignore_names']
    var_21 = bool(var_13['ignore_names'] == ['name1', 'name2'])
    assert var_21 is True
    var_22 = var_13['make_whitelist']
    assert var_22 is True
    var_23 = var_13['min_confidence']
    assert var_23 == 80
    var_24 = var_13['sort_by_size']
    assert var_24 is True
    var_25 = var_13['verbose']
    assert var_25 is True

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["toml_file.py"]\nexclude = ["*test*.py"]\nignore_decorators = ["toml_deco"]\nignore_names = ["toml_name"]\nmake_whitelist = true\nmin_confidence = 70\nsort_by_size = true\nverbose = true\n'
    var_1 = 'cli_file.py'
    var_2 = [var_1]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #13
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 42
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}
    var_6 = module_0._check_input_config(var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_config_tomlfile. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/6 statements.
# Partially parsed test_make_config_verbose_toml_path. Retrieved 1/3 statements.
# Partially parsed test_make_config_invalid_toml_key. Retrieved 1/4 statements.
# Partially parsed test_make_config_invalid_toml_type. Retrieved 1/4 statements.


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
    var_13 = var_0['config']
    assert var_13 == 'pyproject.toml'

import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--exclude=*.py'
    var_3 = '--min-confidence=80'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = var_5['paths']
    var_7 = bool(var_5['paths'] == ['path1', 'path2'])
    assert var_7 is True
    var_8 = var_5['exclude']
    var_9 = bool(var_5['exclude'] == ['*.py'])
    assert var_9 is True
    var_10 = var_5['min_confidence']
    assert var_10 == 80

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["path1", "path2"]\nexclude = ["*.py"]\nmin_confidence = 80\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\npaths = ["path1"]\nmin_confidence = 80\n'
    var_1 = 'path2'
    var_2 = '--min-confidence=90'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '\n[tool.vulture]\nverbose = true\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Data type for min_confidence must be'

import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Please pass at least one file or directory'



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'verbose'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'some_path'
    var_4 = bool(not (var_3 and var_2['verbose']))
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_wrong_type_in_toml. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_verbose_and_toml. Retrieved 3/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
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

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["dir/"]\n    exclude = ["test_*.py"]\n    ignore_decorators = ["@decorator"]\n    ignore_names = ["unused_*"]\n    make_whitelist = true\n    min_confidence = 80\n    sort_by_size = true\n    verbose = true\n    '
    var_1 = '--min-confidence'
    var_2 = '90'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['paths']
    var_3 = bool(var_1['paths'] == [])
    assert var_3 is True
    var_4 = var_1['exclude']
    var_5 = bool(var_1['exclude'] == [])
    assert var_5 is True
    var_6 = var_1['ignore_decorators']
    var_7 = bool(var_1['ignore_decorators'] == [])
    assert var_7 is True
    var_8 = var_1['ignore_names']
    var_9 = bool(var_1['ignore_names'] == [])
    assert var_9 is True
    var_10 = var_1['make_whitelist']
    assert var_10 is False
    var_11 = var_1['min_confidence']
    assert var_11 == 60
    var_12 = var_1['sort_by_size']
    assert var_12 is False
    var_13 = var_1['verbose']
    assert var_13 is False

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
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
    var_0 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
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

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["dir/"]\n    '
    var_1 = '--verbose'
    var_2 = [var_1]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_26. Retrieved 4/17 statements.


def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = 0.5\n'
    var_1 = 'config'
    var_2 = 'verbose'
    var_3 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/9 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--verbose'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['min_confidence']
    assert var_7 == 50
    var_8 = var_6['verbose']
    assert var_8 is True
    var_9 = var_6['paths']
    var_10 = bool(var_6['paths'] == ['path1', 'path2'])
    assert var_10 is True
    var_11 = var_6['exclude']
    var_12 = bool(var_6['exclude'] == [])
    assert var_12 is True
    var_13 = var_6['ignore_decorators']
    var_14 = bool(var_6['ignore_decorators'] == [])
    assert var_14 is True
    var_15 = var_6['ignore_names']
    var_16 = bool(var_6['ignore_names'] == [])
    assert var_16 is True
    var_17 = var_6['make_whitelist']
    assert var_17 is False
    var_18 = var_6['sort_by_size']
    assert var_18 is False
    var_19 = var_6['config']
    assert var_19 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    paths = ["path1"]\n    '
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = var_1['min_confidence']
    assert var_2 == 100
    var_3 = var_1['verbose']
    assert var_3 is False
    var_4 = var_1['make_whitelist']
    assert var_4 is False
    var_5 = var_1['sort_by_size']
    assert var_5 is False
    var_6 = var_1['exclude']
    var_7 = bool(var_1['exclude'] == [])
    assert var_7 is True
    var_8 = var_1['ignore_decorators']
    var_9 = bool(var_1['ignore_decorators'] == [])
    assert var_9 is True
    var_10 = var_1['ignore_names']
    var_11 = bool(var_1['ignore_names'] == [])
    assert var_11 is True
    var_12 = var_1['paths']
    var_13 = bool(var_1['paths'] == [])
    assert var_13 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_toml_path_is_file_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_toml_path_is_file_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_toml_only. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_verbose_and_toml. Retrieved 3/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = 'path/to/file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = var_4['min_confidence']
    assert var_5 == 50
    var_6 = var_4['paths']
    var_7 = bool(var_4['paths'] == ['path/to/file.py'])
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
    var_15 = var_4['sort_by_size']
    assert var_15 is False
    var_16 = var_4['verbose']
    assert var_16 is False
    var_17 = var_4['config']
    assert var_17 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["test_*.py"]\n        min_confidence = 30\n        paths = ["src/"]\n    '
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path/to/file.py'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["test_*.py"]\n        min_confidence = 30\n        paths = ["src/"]\n        verbose = true\n    '
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['min_confidence']
    assert var_3 == 60
    var_4 = var_2['paths']
    var_5 = bool(var_2['paths'] == ['path/to/file.py'])
    assert var_5 is True
    var_6 = var_2['exclude']
    var_7 = bool(var_2['exclude'] == [])
    assert var_7 is True
    var_8 = var_2['ignore_decorators']
    var_9 = bool(var_2['ignore_decorators'] == [])
    assert var_9 is True
    var_10 = var_2['ignore_names']
    var_11 = bool(var_2['ignore_names'] == [])
    assert var_11 is True
    var_12 = var_2['make_whitelist']
    assert var_12 is False
    var_13 = var_2['sort_by_size']
    assert var_13 is False
    var_14 = var_2['verbose']
    assert var_14 is False
    var_15 = var_2['config']
    assert var_15 == 'pyproject.toml'

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        paths = ["src/"]\n    '
    var_1 = '--verbose'
    var_2 = [var_1]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/non/existent/file.toml'



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 0
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_4)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key1'
    var_4 = 'default1'
    var_5 = {var_3: var_4}
    var_6 = module_0._check_input_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default1'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_config_toml_parsing. Retrieved 1/3 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
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
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'file.py'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = var_6['paths']
    var_8 = bool(var_6['paths'] == ['file.py'])
    assert var_8 is True
    var_9 = var_6['exclude']
    var_10 = bool(var_6['exclude'] == ['test_*.py'])
    assert var_10 is True
    var_11 = var_6['min_confidence']
    assert var_11 == 80

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    paths = ["file.py"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    paths = ["file.py"]\n    '
    var_1 = '--exclude'
    var_2 = 'other_*.py'
    var_3 = '--min-confidence'
    var_4 = '90'
    var_5 = [var_1, var_2, var_3, var_4]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key'
    var_1 = 'value'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = 'file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #27
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(not (var_2['detected_toml_path'] and var_2['verbose']))
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_to_false. Retrieved 12/18 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = 'detected_toml_path'
    var_4 = 'verbose'
    var_5 = var_2[var_4]
    var_6 = '[tool.vulture]\nverbose = false'
    var_7 = 'dummy.toml'
    var_8 = 'rb'
    var_9 = open(var_7, var_8)
    var_10 = module_0.make_config(tomlfile=var_9)
    var_11 = var_10[var_4]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #30
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = var_2[var_0]
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_3[var_0]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = bool(var_7 is var_11)
    assert var_12 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_invalid_toml_type. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = var_2['paths']
    var_4 = bool(var_2['paths'] == ['file.py'])
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

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["dir/"]\n    exclude = ["test_*.py"]\n    ignore_decorators = ["@decorator"]\n    ignore_names = ["unused_*"]\n    make_whitelist = true\n    min_confidence = 80\n    sort_by_size = true\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["dir/"]\n    min_confidence = 80\n    '
    var_1 = 'file.py'
    var_2 = '--min-confidence'
    var_3 = '70'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Unknown configuration key'

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_config_with_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_cli_overriding_toml. Retrieved 5/8 statements.


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
    var_0 = '\n        [tool.vulture]\n        paths = ["path1", "path2"]\n        verbose = true\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        paths = ["path1", "path2"]\n        verbose = false\n        min_confidence = 10\n    '
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '20'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_make_config_with_toml_file_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_with_cli_args_overriding_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_invalid_toml_config. Retrieved 1/6 statements.
# Partially parsed test_make_config_with_wrong_type_in_toml. Retrieved 1/6 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = var_5['min_confidence']
    assert var_6 == 50
    var_7 = var_5['paths']
    var_8 = bool(var_5['paths'] == ['path1', 'path2'])
    assert var_8 is True
    var_9 = var_5['exclude']
    var_10 = bool(var_5['exclude'] == [])
    assert var_10 is True
    var_11 = var_5['ignore_decorators']
    var_12 = bool(var_5['ignore_decorators'] == [])
    assert var_12 is True
    var_13 = var_5['ignore_names']
    var_14 = bool(var_5['ignore_names'] == [])
    assert var_14 is True
    var_15 = var_5['make_whitelist']
    assert var_15 is False
    var_16 = var_5['sort_by_size']
    assert var_16 is False
    var_17 = var_5['verbose']
    assert var_17 is False

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = 10\n        paths = ["path1"]\n    '
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path2'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-arg'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_a_number'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_a_number"\n    '



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 4/7 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 2/6 statements.
# Partially parsed test_make_config_with_invalid_toml_type. Retrieved 2/6 statements.


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
    var_0 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_1 = []
    var_2 = bool(False)
    assert var_2 is True

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test.toml'
    var_1 = '[tool.vulture]\nmin_confidence = 0.5'



# Parsed testcases at query #38
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = var_2[var_0]
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = bool(var_8 is var_12)
    assert var_13 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_make_config_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True

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
    var_0 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'

def test_case_0():
    var_0 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\n'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/9 statements.


def test_case_0():
    var_0 = b'[vulture]\nmin_confidence = 0.5'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'nonexistent_file.toml'



