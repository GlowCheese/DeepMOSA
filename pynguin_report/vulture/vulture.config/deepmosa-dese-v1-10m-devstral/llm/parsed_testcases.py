####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_config_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_verbose_toml_output. Retrieved 1/4 statements.
# Partially parsed test_make_config_invalid_toml_key. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = '--verbose'
    var_3 = 'src/'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["src/"]\n    min_confidence = 80\n    verbose = true\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["src/"]\n    min_confidence = 80\n    verbose = false\n    '
    var_1 = '--min-confidence'
    var_2 = '90'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["src/"]\n    verbose = true\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = '--invalid-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 456
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_verbose_output_when_toml_detected. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'rb'
    var_2 = '--verbose'
    var_3 = [var_2]



# Parsed testcases at query #4
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_input_config_correct_type. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 0
    var_6 = ''
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = var_4[var_0]
    var_9 = var_7[var_0]
    var_10 = var_4[var_1]
    var_11 = var_7[var_1]



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_toml_path_is_file.




# Parsed testcases at query #7
#--------------------------

# Failed to parse test_check_input_config_with_correct_types.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'valid_file.toml'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/9 statements.
# Partially parsed test_make_config_verbose_with_toml. Retrieved 1/4 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    '
    var_1 = '--exclude'
    var_2 = 'other_*.py'
    var_3 = '--min-confidence'
    var_4 = '70'
    var_5 = [var_1, var_2, var_3, var_4]

import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    verbose = true\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_verbose_output_when_toml_detected. Retrieved 5/7 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true\n'
    var_1 = '--config'
    var_2 = 'pyproject.toml'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test.toml'



# Parsed testcases at query #14
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

import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 0
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 1/5 statements.
# Partially parsed test_make_config_with_invalid_toml_type. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)

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

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'verbose'
    var_1 = 'test.toml'
    var_2 = '--verbose'
    var_3 = [var_2]
    var_4 = make_config(tomlfile=var_1, argv=var_3)[var_0]
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 5/8 statements.
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

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py"]\n    ignore_decorators = ["deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1"]\n    '
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = 'path2'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py"]\n    ignore_decorators = ["deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1"]\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #20
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 2
    var_6 = 'another'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0._check_input_config(var_7)



# Parsed testcases at query #21
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #22
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0._check_input_config(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test.toml'



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

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 4/7 statements.
# Partially parsed test_make_config_verbose_toml_output. Retrieved 1/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 30\n    '
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = [var_1, var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    verbose = true\n    '

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



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



# Parsed testcases at query #2
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = 456
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_config_with_toml_and_cli_args. Retrieved 4/6 statements.
# Partially parsed test_make_config_with_invalid_toml_key. Retrieved 2/5 statements.
# Partially parsed test_make_config_with_invalid_toml_type. Retrieved 2/5 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 10\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '
    var_1 = '--verbose'
    var_2 = 'path3'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_1 = []

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_1 = []

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



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
    var_7 = 2
    var_8 = 'another_value'
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)



# Parsed testcases at query #5
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 42
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'default1'
    var_8 = 0
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_config_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 6/9 statements.
# Partially parsed test_make_config_verbose_toml_output. Retrieved 1/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--help'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = '--verbose'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    paths = ["path1"]\n    '
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = '--verbose'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]

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

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["path1"]\n    '



# Parsed testcases at query #7
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #8
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._parse_args(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = '*.py,test_*.py'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-decorators'
    var_1 = '@app.route,@require_*'
    var_2 = [var_0, var_1]
    var_3 = module_0._parse_args(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--ignore-names'
    var_1 = 'visit_*,do_*'
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
    var_0 = 'file.py'
    var_1 = '--exclude'
    var_2 = '*.py'
    var_3 = '--ignore-decorators'
    var_4 = '@app.route'
    var_5 = '--min-confidence'
    var_6 = '90'
    var_7 = '--verbose'
    var_8 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0._parse_args(var_8)

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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'valid.toml'



# Parsed testcases at query #10
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_config_toml_only. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--verbose'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)

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

import vulture.config as module_0

def test_case_0():
    var_0 = '--unknown-key'
    var_1 = 'value'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_a_number'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test.toml'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_config_toml_overrides. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 12/15 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

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
    var_11 = 'path1'
    var_12 = 'path2'
    var_13 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12]
    var_14 = module_0.make_config(var_13)

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["test_*", "*.pyc"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 50\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '

def test_case_0():
    var_0 = '\n        [tool.vulture]\n        exclude = ["test_*", "*.pyc"]\n        ignore_decorators = ["deco1", "deco2"]\n        ignore_names = ["name1", "name2"]\n        make_whitelist = true\n        min_confidence = 50\n        sort_by_size = true\n        verbose = true\n        paths = ["path1", "path2"]\n    '
    var_1 = '--exclude'
    var_2 = 'override_*,*.pyc'
    var_3 = '--ignore-decorators'
    var_4 = 'override_deco'
    var_5 = '--ignore-names'
    var_6 = 'override_name'
    var_7 = '--min-confidence'
    var_8 = '90'
    var_9 = 'override_path1'
    var_10 = 'override_path2'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

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



# Parsed testcases at query #15
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = 'default_value'
    var_4 = {var_0: var_3}
    var_5 = module_0._check_input_config(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_verbose_toml_path_print. Retrieved 3/5 statements.


def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = '[tool.vulture]\nverbose = true'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_toml_path_is_file. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'pyproject.toml'



# Parsed testcases at query #18
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'default1'
    var_6 = {var_0: var_5, var_1: var_3}
    var_7 = module_0._check_input_config(var_4)

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



# Parsed testcases at query #19
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_toml_path_is_file.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_config_with_toml_file. Retrieved 1/4 statements.
# Partially parsed test_make_config_cli_overrides_toml. Retrieved 5/8 statements.


import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'test_*.py'
    var_2 = '--verbose'
    var_3 = 'src/'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    verbose = true\n    paths = ["src/"]\n    '

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    verbose = false\n    '
    var_1 = '--exclude'
    var_2 = 'temp_*.py'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'existing_file.toml'



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
    var_7 = 2
    var_8 = 'another'
    var_9 = False
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = module_0._check_input_config(var_10)



# Parsed testcases at query #25
#--------------------------




import vulture.config as module_0

def test_case_0():
    var_0 = '--verbose'
    var_1 = [var_0]
    var_2 = 'test.toml'
    var_3 = module_0.make_config(var_1, var_2)



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_toml_path_is_file.




