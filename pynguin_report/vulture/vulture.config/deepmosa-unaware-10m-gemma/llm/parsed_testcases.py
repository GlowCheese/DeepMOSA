####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'utf-8'
    var_1 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing no paths raises an InputError.'
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = 'Test that providing the wrong type in TOML raises an InputError.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'path/to/code'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'Test that providing an unknown key in TOML raises an InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = true'
    var_2 = 'path/to/code'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'Test that make_config correctly loads from an actual file on disk.'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.vulture]\nmin_confidence = 25\nverbose = true'
    var_3 = 'utf-8'
    var_4 = 'path/to/code'
    var_5 = '--config'



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'vulture'
    var_2 = 'path1'
    var_3 = '--exclude'
    var_4 = 'a.py,b.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--sort-by-size'
    var_8 = '--verbose'
    var_9 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config()
    var_11 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["from_toml.py"]\nignore_names = ["unused_var"]\n'
    var_12 = 'pyproject.toml'
    var_13 = 'some_path'
    var_14 = '80'
    var_15 = [var_1, var_13, var_5, var_14]
    var_16 = module_0.make_config()
    var_17 = b'[tool.vulture]\nverbose = true\npaths = ["stream_path"]'
    var_18 = module_0.make_config()
    var_19 = b'[tool.vulture]\nunknown_key = true'
    var_20 = str(var_18)
    var_21 = 'paths'
    var_22 = []
    var_23 = {var_21: var_22}
    var_24 = module_0._check_output_config(var_23)
    var_25 = str(var_21)
    var_26 = b'[tool.vulture]\nmin_confidence = "high"'
    var_27 = str(var_21)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = b'\n[tool.vulture]\nmin_confidence = 50\nverbose = true\npaths = ["from_toml"]\n'
    var_4 = 'pyproject.toml'
    var_5 = '--config'
    var_6 = '--min-confidence'
    var_7 = '80'
    var_8 = 'extra_path'
    var_9 = module_0.make_config(var_1)
    var_10 = '--min-confidence'
    var_11 = 'not_an_int'
    var_12 = [var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = 'unknown_key'
    var_15 = True
    var_16 = {var_14: var_15}
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = b'[tool.vulture]\npaths = ["io_path"]\nmin_confidence = 10'
    var_20 = 'some_path'
    var_21 = [var_20]



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = 'pyproject.toml'
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test.py"]\npaths = ["."]\n'
    var_5 = 'pyproject.toml'
    var_6 = '--min-confidence'
    var_7 = '80'
    var_8 = 'src/'
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = '--exclude'
    var_12 = 'ignore1.py,ignore2.py'
    var_13 = [var_11, var_12, var_8]
    var_14 = module_0.make_config(var_13)
    var_15 = 'min_confidence'
    var_16 = 'not_an_int'
    var_17 = {var_15: var_16}
    var_18 = module_0._check_input_config(var_17)
    var_19 = 'unknown_key'
    var_20 = True
    var_21 = {var_19: var_20}
    var_22 = module_0._check_input_config(var_21)
    var_23 = b'[tool.vulture]\nverbose = true\npaths = ["path1"]'
    var_24 = 'path2'
    var_25 = [var_24]
    var_26 = '--sort-by-size'
    var_27 = [var_26, var_8]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = [var_29, var_8]
    var_31 = module_0.make_config(var_30)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = '[tool.vulture]\nmin_confidence = 50\nverbose = true\npaths = ["test_dir"]'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--paths'
    var_5 = 'src/'
    var_6 = '--config'
    var_7 = '--exclude'
    var_8 = 'venv/,.git/'
    var_9 = 'my_folder'
    var_10 = [var_7, var_8, var_4, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = b'[tool.vulture]\nignore_names = ["foo", "bar"]\npaths = ["path1"]'
    var_13 = []
    var_14 = '--min-confidence'
    var_15 = 'not_an_int'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = 'min_confidence'
    var_19 = 'high'
    var_20 = {var_18: var_19}
    var_21 = module_0._check_input_config(var_20)
    var_22 = '--config'
    var_23 = [var_22, var_19]
    var_24 = module_0.make_config(var_23)
    var_25 = b'[tool.vulture]\nmin_confidence=10'
    var_26 = []
    var_27 = 'invalid_key'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = module_0._check_input_config(var_29)

def test_case_0():
    var_0 = 'Test that invalid types in TOML trigger InputError.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = []



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = 'utf-8'
    var_3 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that InputError is raised if no paths are provided.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that the function attempts to read from a real file when no tomlfile is passed.'
    var_1 = b'[tool.vulture]\nmin_confidence = 25\n'
    var_2 = 'tool'
    var_3 = 'vulture'
    var_4 = 'min_confidence'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'some_path'
    var_9 = [var_8]
    var_10 = module_0.make_config(var_9)

def test_case_0():
    var_0 = 'Test that passing the wrong type via CLI (if it were possible) or TOML raises error.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'some_path'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'Test that unknown configuration keys raise InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = True'
    var_2 = 'some_path'
    var_3 = [var_2]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the merging logic of CLI and TOML.'
    var_1 = 'utf-8'

def test_case_0():
    var_0 = 'Test that _check_output_config raises error if no paths are provided.'
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = [var_1, var_2]
    var_4 = b''

def test_case_0():
    var_0 = 'Test that the function correctly attempts to read from a real file if no tomlfile arg is passed.'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.vulture]\nmin_confidence = 80'
    var_3 = 'utf-8'
    var_4 = '--config'
    var_5 = '--paths'
    var_6 = 'test_dir'

def test_case_0():
    var_0 = 'Test that providing the wrong type in TOML raises InputError.'
    var_1 = '[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'utf-8'
    var_3 = []

def test_case_0():
    var_0 = 'Test that providing an unknown key in TOML raises InputError.'
    var_1 = '[tool.vulture]\nunknown_key = True'
    var_2 = 'utf-8'
    var_3 = []

def test_case_0():
    var_0 = 'Test that the CSV parsing logic in _parse_args works for excludes.'
    var_1 = '--exclude'
    var_2 = 'file1.py,file2.py'
    var_3 = '--paths'
    var_4 = 'src'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = b''



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = 'path/to/code'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test_*.py"]\nverbose = true\n'
    var_8 = '80'
    var_9 = 'src/'
    var_10 = [var_0, var_8, var_9]
    var_11 = '--min-confidence'
    var_12 = '10'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '--min-confidence'
    var_16 = 'not_an_int'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = b'[tool.vulture]\nunknown_key = true'
    var_20 = []
    var_21 = 'pyproject.toml'
    var_22 = '[tool.vulture]\nmin_confidence = 10'
    var_23 = 'utf-8'
    var_24 = 'some_path'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #9
#--------------------------


import builtins as module_0
import vulture.config as module_1

def test_case_0():
    var_0 = 'test_path'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = module_0.object()
    var_4 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test*.py"]\nverbose = true\n'
    var_5 = 'pyproject.toml'
    var_6 = 'paths'
    var_7 = 'min_confidence'
    var_8 = 'config'
    var_9 = 'src'
    var_10 = [var_9]
    var_11 = 80
    var_12 = 'src'
    var_13 = [var_12]
    var_14 = module_1.make_config(var_13)
    var_15 = [var_9]
    var_16 = 90
    var_17 = 'src'
    var_18 = [var_17]
    var_19 = module_1.make_config(var_18)
    var_20 = []
    var_21 = []
    var_22 = module_1.make_config(var_21)
    var_23 = 'direct_stream'
    var_24 = [var_23]
    var_25 = {var_6: var_24}
    var_26 = 'only_path'
    var_27 = [var_26]
    var_28 = {var_6: var_27}
    var_29 = 'only_path'
    var_30 = [var_29]
    var_31 = module_1.make_config(var_30)
    var_32 = [var_9]
    var_33 = 'not_an_int'
    var_34 = {var_6: var_32, var_7: var_33}
    var_35 = 'src'
    var_36 = [var_35]
    var_37 = module_1.make_config(var_36)

import vulture.config as module_0

def test_case_0():
    var_0 = 'paths'
    var_1 = 'unknown_key'
    var_2 = 'src'
    var_3 = [var_2]
    var_4 = 'invalid'
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = 'src'
    var_7 = [var_6]
    var_8 = module_0.make_config(var_7)



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = 'path/to/code'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test*.py"]\nverbose = true\n'
    var_8 = 'pyproject.toml'
    var_9 = '80'
    var_10 = 'some_dir'
    var_11 = [var_0, var_9, var_10]
    var_12 = None
    var_13 = module_0.make_config(var_11, var_12)
    var_14 = b'[tool.vulture]\nignore_names = ["unused"]'
    var_15 = '--ignore-names'
    var_16 = 'extra_name'
    var_17 = 'path'
    var_18 = [var_15, var_16, var_17]
    var_19 = '--min-confidence'
    var_20 = 'not_an_int'
    var_21 = 'path'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = b'[tool.vulture]\nunknown_key = true'
    var_25 = 'path'
    var_26 = [var_25]
    var_27 = []
    var_28 = module_0.make_config(var_27)
    var_29 = b'[tool.vulture]\nmin_confidence = "high"'
    var_30 = 'path'
    var_31 = [var_30]



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    pass

import vulture.config as module_0

def test_case_0():
    var_0 = '\n    A comprehensive test for make_config covering:\n    - CLI precedence over TOML\n    - Default value application\n    - Error handling for empty paths\n    '
    var_1 = 'tool'
    var_2 = 'vulture'
    var_3 = 'min_confidence'
    var_4 = 'exclude'
    var_5 = 'paths'
    var_6 = 20
    var_7 = 'temp/*'
    var_8 = [var_7]
    var_9 = 'src'
    var_10 = [var_9]
    var_11 = {var_3: var_6, var_4: var_8, var_5: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = '--min-confidence'
    var_15 = '80'
    var_16 = 'my_dir'
    var_17 = [var_14, var_15, var_16]
    var_18 = b'dummy'
    var_19 = module_0.make_config(var_17, var_1)
    var_20 = []
    var_21 = b'dummy'
    var_22 = module_0.make_config(var_20, var_1)
    var_23 = 'not_an_int'
    var_24 = [var_14, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = 'unknown_key'
    var_27 = 'value'
    var_28 = [var_9]
    var_29 = {var_26: var_27, var_5: var_28}
    var_30 = {var_22: var_29}
    var_31 = {var_1: var_30}
    var_32 = 'src'
    var_33 = [var_32]
    var_34 = b'dummy'
    var_35 = module_0.make_config(var_33, var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test behavior when no TOML file is found on disk.'
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = 'path/to/code'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = 'path/to/code'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nmin_confidence = 10\nexclude = ["test_*.py"]\nverbose = true\n'
    var_7 = 'pyproject.toml'
    var_8 = '--config'
    var_9 = 'some_path'
    var_10 = [var_8, var_1, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = '80'
    var_13 = 'some_path'
    var_14 = [var_8, var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = b'[tool.vulture]\nmin_confidence = "high"]'
    var_17 = 'path'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = 'pyproject.toml'
    var_2 = 0
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = False
    var_7 = False
    var_8 = False
    var_9 = []
    var_10 = module_0.make_config(var_9)

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'file1.py,file2.py'
    var_2 = '--ignore-names'
    var_3 = 'name1,name2'
    var_4 = 'path'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/dir'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test*.py"]\nverbose = true\n'
    var_7 = 'pyproject.toml'
    var_8 = 'some_path'
    var_9 = '80'
    var_10 = [var_8, var_1, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = b'[tool.vulture]\nignore_names = ["unused"]'
    var_13 = '.'
    var_14 = [var_13]
    var_15 = b'[tool.vulture]\nmin_confidence = "high"]'
    var_16 = '.'
    var_17 = [var_16]
    var_18 = 'Namespace'
    var_19 = ()
    var_20 = 'paths'
    var_21 = 'exclude'
    var_22 = 'ignore_decorators'
    var_23 = 'ignore_names'
    var_24 = 'make_whitelist'
    var_25 = 'sort_by_size'
    var_26 = 'verbose'
    var_27 = 'config'
    var_28 = 'min_confidence'
    var_29 = []
    var_30 = []
    var_31 = []
    var_32 = []
    var_33 = False
    var_34 = 'pyproject.toml'
    var_35 = {var_20: var_29, var_21: var_30, var_22: var_31, var_23: var_32, var_24: var_33, var_25: var_33, var_26: var_33, var_27: var_34, var_28: var_33}
    var_36 = []
    var_37 = module_0.make_config(var_36)
    var_38 = str(var_31)

def test_case_0():
    var_0 = b'[tool.vulture]\nunknown_key = True'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = str(var_2)



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path/to/code'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--sort-by-size'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = 'src'
    var_10 = '20'
    var_11 = [var_9, var_3, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n[tool.vulture]\nmin_confidence = 10\nexclude = ["test*.py"]\nsort_by_size = true\n'
    var_14 = 'pyproject.toml'
    var_15 = '80'
    var_16 = [var_9, var_3, var_15]
    var_17 = b'\n[tool.vulture]\nmin_confidence = "not-an-int"\n'
    var_18 = 'src'
    var_19 = [var_18]
    var_20 = b'\n[tool.vulture]\nunknown_key = True\n'
    var_21 = 'src'
    var_22 = [var_21]
    var_23 = '--exclude'
    var_24 = 'a.py,b.py'
    var_25 = '--ignore-names'
    var_26 = 'func1,func2'
    var_27 = [var_9, var_23, var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = '-v'
    var_31 = [var_9, var_29, var_30]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #15
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nexclude = ["*.tmp"]\nmin_confidence = 20\nverbose = true\npaths = ["toml_path"]\n'
    var_7 = 'pyproject.toml'
    var_8 = '80'
    var_9 = [var_1, var_8]
    var_10 = module_0.make_config(var_9, var_0)
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = '--min-confidence'
    var_14 = 'not_an_int'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #16
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_dir/'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test.py"]\nverbose = true\n'
    var_4 = 'pyproject.toml'
    var_5 = '--config'
    var_6 = '--sort-by-size'
    var_7 = '--min-confidence'
    var_8 = '80'
    var_9 = 'test_dir/'
    var_10 = '--min-confidence'
    var_11 = 'not_an_int'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n[tool.vulture]\nunknown_key = True\n'
    var_15 = 'invalid.toml'
    var_16 = 'test_dir/'
    var_17 = '--config'
    var_18 = [var_16, var_17, var_11]
    var_19 = module_0.make_config(var_18)
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = b'[tool.vulture]\nignore_names = ["foo"]'
    var_23 = [var_20]
    var_24 = '--ignore-decorators'
    var_25 = 'deco1,deco2,deco3'
    var_26 = [var_20, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--make-whitelist'
    var_29 = [var_20, var_28]
    var_30 = module_0.make_config(var_29)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'utf-8'
    var_1 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that missing paths raises InputError.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

def test_case_0():
    var_0 = 'Test that invalid type in TOML raises InputError.'
    var_1 = '[tool.vulture]\nmin_confidence = "not_an_int"'
    var_2 = 'utf-8'
    var_3 = 'path/to/code'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test that unknown keys in TOML raise InputError.'
    var_1 = '[tool.vulture]\nunknown_key = true'
    var_2 = 'utf-8'
    var_3 = 'path/to/code'
    var_4 = [var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test loading from actual file path via mocking open.'
    var_1 = '[tool.vulture]\nmin_confidence = 25'
    var_2 = 'tool'
    var_3 = 'vulture'
    var_4 = 'min_confidence'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'path/to/code'
    var_9 = '--config'
    var_10 = 'fake_pyproject.toml'
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path/'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--sort-by-size'
    var_6 = '--exclude'
    var_7 = 'test.py,temp/'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n[tool.vulture]\nmin_confidence = 25\nignore_names = ["unused_var"]\npaths = ["src/"]\n'
    var_11 = 'pyproject.toml'
    var_12 = []
    var_13 = '--min_confidence'
    var_14 = '80'
    var_15 = [var_13, var_14]
    var_16 = '--min-confidence'
    var_17 = 'not_an_int'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = 'invalid int value'
    var_21 = str(var_17)
    var_22 = var_20 in var_19
    var_23 = 'argument'
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '.'
    var_27 = [var_26]
    var_28 = 'pyproject.toml'
    var_29 = '--ignore-decorators'
    var_30 = 'deco1,deco2'
    var_31 = 'path/'
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = []
    var_4 = module_0.make_config(var_3)
    var_5 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["*.tmp"]\n'
    var_6 = 'pyproject.toml'
    var_7 = 'test_dir'
    var_8 = '--min-confidence'
    var_9 = '80'
    var_10 = '--exclude'
    var_11 = 'pattern1,pattern2'
    var_12 = [var_7, var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = 'test_dir'
    var_15 = '--min-confidence'
    var_16 = 'not_an_int'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = b'\n[tool.vulture]\nunknown_key = True\n'
    var_20 = 'invalid.toml'
    var_21 = 'test_dir'
    var_22 = [var_21]
    var_23 = '--config'
    var_24 = [var_23, var_22]
    var_25 = module_0.make_config(var_24)
    var_26 = []
    var_27 = 'pyproject.toml'
    var_28 = []
    var_29 = module_0.make_config(var_28)
    var_30 = '--ignore-names'
    var_31 = 'func1,func2'
    var_32 = [var_24, var_30, var_31]
    var_33 = 'test_dir'
    var_34 = [var_33]
    var_35 = 'pyproject.toml'
    var_36 = 'func1'
    var_37 = 'func2'
    var_38 = [var_36, var_37]
    var_39 = module_0.make_config(var_32)
    var_40 = '--sort-by-size'
    var_41 = '--verbose'
    var_42 = [var_35, var_40, var_41]
    var_43 = 'test_dir'
    var_44 = [var_43]
    var_45 = 'pyproject.toml'
    var_46 = True
    var_47 = module_0.make_config(var_42)



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test*.py"]\nverbose = true\npaths = ["src"]\n'
    var_1 = 'pyproject.toml'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'some_path.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '--exclude'
    var_8 = 'extra.py'
    var_9 = '--sort-by-size'
    var_10 = 'dir/'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = 'path/to/code'
    var_14 = [var_13]
    var_15 = '--min-confidence'
    var_16 = 'not_an_int'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = b'[tool.vulture]\nunknown_key = true'
    var_20 = 'path'
    var_21 = [var_20]
    var_22 = 'pyproject.toml'
    var_23 = 0
    var_24 = []
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = False
    var_29 = False
    var_30 = False
    var_31 = []
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path/to/dir'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test_*.py"]\nsort_by_size = true\n'
    var_9 = 'pyproject.toml'
    var_10 = 'src/'
    var_11 = '80'
    var_12 = [var_10, var_3, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--exclude'
    var_15 = 'pattern1,pattern2'
    var_16 = '--ignore-names'
    var_17 = 'name1'
    var_18 = [var_10, var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = 'src/'
    var_21 = '--min-confidence'
    var_22 = 'not_an_int'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = b'\n[tool.vulture]\nunknown_key = "error"\n'
    var_26 = 'bad.toml'
    var_27 = 'src/'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28, var_22)
    var_30 = []
    var_31 = module_0.make_config(var_30)
    var_32 = str(var_30)
    var_33 = '--make-whitelist'
    var_34 = '--sort-by-size'
    var_35 = [var_10, var_33, var_34]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #6
#--------------------------


import builtins as module_0
import vulture.config as module_1

def test_case_0():
    var_0 = []
    var_1 = 'pyproject.toml'
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = module_0.object()
    var_5 = module_0.object()
    var_6 = module_0.object()
    var_7 = module_0.object()
    var_8 = module_0.object()
    var_9 = []
    var_10 = module_1.make_config(var_9)
    var_11 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test*.py"]\nverbose = true\n'
    var_12 = 'pyproject.toml'
    var_13 = 'path/to/code'
    var_14 = '--min-confidence'
    var_15 = '80'
    var_16 = '--sort-by-size'
    var_17 = [var_13, var_14, var_15, var_16]
    var_18 = 'path/to/code'
    var_19 = '--min-confidence'
    var_20 = '80'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_1.make_config(var_21)
    var_23 = 'path/to/code'
    var_24 = '--min-confidence'
    var_25 = 'not_an_int'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_1.make_config(var_26)
    var_28 = 'some_path'
    var_29 = [var_28]
    var_30 = b'[tool.vulture]\nunknown_key = true'
    var_31 = 'some_path'
    var_32 = [var_31]

def test_case_0():
    var_0 = 'Specific test for precedence: CLI > TOML > Defaults'
    var_1 = b'[tool.vulture]\nmin_confidence = 20\nverbose = false'
    var_2 = 'test_config.toml'
    var_3 = 'my_dir'
    var_4 = '--min-confidence'
    var_5 = '90'
    var_6 = '--verbose'
    var_7 = [var_3, var_4, var_5, var_6]



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path/to/code'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--sort-by-size'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '--exclude'
    var_9 = 'test.py,venv/'
    var_10 = [var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["old.py"]\nverbose = true\n'
    var_13 = 'pyproject.toml'
    var_14 = '--min-confidence'
    var_15 = '40'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--sort-by-size'
    var_19 = [var_18]
    var_20 = '--min-confidence'
    var_21 = 'not-an-int'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = str(var_6)
    var_25 = b'[tool.vulture]\nmin_confidence = "high"'
    var_26 = []
    var_27 = b'[tool.vulture]\nunknown_key = true'
    var_28 = []
    var_29 = []
    var_30 = module_0.make_config(var_29)
    var_31 = str(var_29)
    var_32 = 'path/to/dir'
    var_33 = '--ignore-decorators'
    var_34 = 'deco1,deco2'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'some_path.py'
    var_1 = [var_0]
    var_2 = 'utf-8'

def test_case_0():
    var_0 = '{"tool": {"vulture": {"min_confidence": 10, "paths": ["file1.py"]}}}'
    var_1 = 'utf-8'
    var_2 = '--min-confidence'
    var_3 = '20'
    var_4 = 'path/to/dir'
    var_5 = [var_2, var_3, var_4]

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '{"tool": {"vulture": {"min_confidence": "not_an_int"}}}'
    var_1 = 'utf-8'
    var_2 = 'some_path.py'
    var_3 = [var_2]

def test_case_0():
    var_0 = '{"tool": {"vulture": {"unknown_key": True}}}'
    var_1 = 'utf-8'
    var_2 = 'some_path.py'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'some_path.py'
    var_1 = [var_0]
    var_2 = b'{"tool": {"vulture": {}}}'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Tests the merging logic of CLI and TOML configurations.'
    var_1 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that an error is raised if no paths are provided in either source.'
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = 'Tests that providing the wrong type via CLI raises InputError.'
    var_1 = 'min_confidence'
    var_2 = 'not_an_int'
    var_3 = (var_1, var_2)
    var_4 = 'paths'
    var_5 = 'p'
    var_6 = [var_5]
    var_7 = (var_4, var_6)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that make_config correctly reads from an existing file on disk.'
    var_1 = b'[tool.vulture]\nmin_confidence = 25\npaths = ["src"]'
    var_2 = 'tool'
    var_3 = 'vulture'
    var_4 = 'min_confidence'
    var_5 = 'paths'
    var_6 = 25
    var_7 = 'src'
    var_8 = [var_7]
    var_9 = {var_4: var_6, var_5: var_8}
    var_10 = {var_3: var_9}
    var_11 = '--verbose'
    var_12 = [var_11]
    var_13 = module_0.make_config(var_12)

def test_case_0():
    var_0 = 'Tests that an unknown key in TOML raises InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = "value"\npaths = ["src"]'
    var_2 = '--paths'
    var_3 = 'src'
    var_4 = [var_2, var_3]



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test*.py"]\nverbose = true\n'
    var_1 = 'pyproject.toml'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--paths'
    var_5 = 'src'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '--exclude'
    var_9 = 'temp.py'
    var_10 = '.'
    var_11 = [var_8, var_9, var_4, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = 'test_dir'
    var_14 = [var_4, var_13]
    var_15 = '--min-confidence'
    var_16 = '10'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--min-confidence'
    var_20 = 'not_an_int'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = b'[tool.vulture]\nunknown_key = true'
    var_24 = '--paths'
    var_25 = '.'
    var_26 = [var_24, var_25]
    var_27 = b'[tool.vulture]\nmin_confidence = "high"'
    var_28 = '--paths'
    var_29 = '.'
    var_30 = [var_28, var_29]



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_path'
    var_1 = [var_0]
    var_2 = 'pyproject.toml'
    var_3 = 0
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = False
    var_8 = False
    var_9 = False
    var_10 = 'test_path'
    var_11 = [var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = b'[tool.vulture]\nmin_confidence = 50\nverbose = false\npaths = ["toml_path"]'
    var_14 = 'pyproject.toml'
    var_15 = 'path1'
    var_16 = [var_15]
    var_17 = 80
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = False
    var_22 = True
    var_23 = 'path1'
    var_24 = [var_23]
    var_25 = []
    var_26 = 'pyproject.toml'
    var_27 = 0
    var_28 = []
    var_29 = []
    var_30 = []
    var_31 = False
    var_32 = False
    var_33 = False
    var_34 = []
    var_35 = module_0.make_config(var_34)
    var_36 = 'path'
    var_37 = [var_36]
    var_38 = 'pyproject.toml'
    var_39 = 'high'
    var_40 = []
    var_41 = []
    var_42 = []
    var_43 = False
    var_44 = 'path'
    var_45 = [var_44]
    var_46 = module_0.make_config(var_45)
    var_47 = 'path'
    var_48 = [var_47]
    var_49 = 'pyproject.toml'
    var_50 = 0
    var_51 = []
    var_52 = []
    var_53 = []
    var_54 = False
    var_55 = False
    var_56 = False
    var_57 = 'error'



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = 'cli_path.py'
    var_5 = [var_4]
    var_6 = 'exclude1,exclude2'
    var_7 = [var_6]
    var_8 = 50
    var_9 = True
    var_10 = 'dummy.toml'
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["file1.py"]\npaths = ["toml_path.py"]\n'
    var_16 = 'pyproject.toml'
    var_17 = 'cli_override.py'
    var_18 = [var_17]
    var_19 = 80
    var_20 = '--min-confidence'
    var_21 = '80'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = 'test.py'
    var_25 = [var_24]
    var_26 = 'not_an_int'
    var_27 = []
    var_28 = []
    var_29 = module_0.make_config(var_28)
    var_30 = b'\n[tool.vulture]\nunknown_key = "oops"\npaths = ["test.py"]\n'
    var_31 = 'test.py'
    var_32 = [var_31]
    var_33 = 'dummy.toml'

def test_case_0():
    var_0 = b'\n[tool.vulture]\npaths = ["test.py"]\nverbose = true\n'
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = 'dummy.toml'
    var_4 = True



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/code'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = 'test_dir'
    var_8 = '20'
    var_9 = '--verbose'
    var_10 = [var_7, var_1, var_8, var_9]
    var_11 = module_0.make_config(var_10, var_5)
    var_12 = b'\n[tool.vulture]\nmin_confidence = 10\nexclude = ["test*.py"]\nsort_by_size = true\npaths = ["from_toml"]\n'
    var_13 = 'pyproject.toml'
    var_14 = 'from_toml'
    var_15 = [var_14]
    var_16 = b'\n[tool.vulture]\nmin_confidence = 10\nexclude = ["old_pattern"]\npaths = ["original_path"]\n'
    var_17 = 'new_path'
    var_18 = '80'
    var_19 = [var_17, var_1, var_18]
    var_20 = 'path'
    var_21 = '--min-confidence'
    var_22 = 'not_an_int'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = b'[tool.vulture]\nunknown_key = True\npaths = ["p"]'
    var_26 = 'p'
    var_27 = [var_26]
    var_28 = b'[tool.vulture]\nmin_confidence=1'
    var_29 = []
    var_30 = module_0.make_config(var_29, var_22)



