####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["temp/"]\npaths = ["src/"]\n'
    var_13 = 'pyproject.toml'
    var_14 = 'cli_path'
    var_15 = [var_14]
    var_16 = 80
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = False
    var_21 = True
    var_22 = '--min-confidence'
    var_23 = '80'
    var_24 = [var_22, var_23, var_14]
    var_25 = module_0.make_config(var_24)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'utf-8'
    var_1 = b''

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that missing paths raises InputError.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that invalid type in CLI raises InputError.'
    var_1 = '--min-confidence'
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = 'Test that unknown keys in TOML raise InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = true'
    var_2 = 'path/to/dir'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that DEFAULTS are applied when nothing is provided in CLI or TOML.'
    var_1 = 'some_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Basic functional tests for merging and precedence.'
    var_1 = {}
    var_2 = 'tool'
    var_3 = {}
    var_4 = 'vulture'
    var_5 = {}
    var_6 = None
    var_7 = any(var_2)

def test_case_0():
    var_0 = 'Test specifically using the tomlfile argument.'
    var_1 = b'[tool.vulture]\nmin_confidence = 25\npaths = ["src"]'
    var_2 = 'src'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that missing paths raises InputError.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that invalid types in CLI raise InputError.'
    var_1 = '--min-confidence'
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = 'Test that unknown keys in TOML raise InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = true'
    var_2 = 'src'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Integration test: testing the actual file reading logic.'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.vulture]\nmin_confidence = 10\nverbose = true'
    var_3 = 'utf-8'
    var_4 = '--config'
    var_5 = 'my_dir'
    var_6 = [var_4, var_1, var_5]
    var_7 = module_0.make_config(var_6)



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = 'some_dir'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--sort-by-size'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = b'\n[tool.vulture]\nexclude = ["test.py"]\nmin_confidence = 10\nverbose = true\n'
    var_8 = 'pyproject.toml'
    var_9 = 'my_folder'
    var_10 = '90'
    var_11 = [var_0, var_9, var_2, var_10]
    var_12 = module_0.make_config(var_11, var_0)
    var_13 = 'vulture'
    var_14 = '--min-confidence'
    var_15 = 'not_an_int'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = b'[tool.vulture]\nunknown_key = true'
    var_19 = 'vulture'
    var_20 = '.'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21, var_16)
    var_23 = 'vulture'
    var_24 = '--config'
    var_25 = 'non_existent.toml'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = []
    var_29 = module_0.make_config(var_28)
    var_30 = str(var_28)
    var_31 = '.'
    var_32 = '--exclude'
    var_33 = 'file1.py,file2.py'
    var_34 = '--ignore-names'
    var_35 = 'func_a,func_b'
    var_36 = [var_28, var_31, var_32, var_33, var_34, var_35]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #5
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
    var_10 = [var_9, var_3, var_4]
    var_11 = module_0.make_config(var_10)
    var_12 = b'\n[tool.vulture]\nexclude = ["*.tmp"]\nmin_confidence = 20\npaths = ["file1.py"]\n'
    var_13 = 'pyproject.toml'
    var_14 = '--config'
    var_15 = '80'
    var_16 = module_0.make_config(var_7)
    var_17 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = 'invalid.toml'
    var_19 = '--config'
    var_20 = 'src'
    var_21 = [var_19, var_3, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = b'\n[tool.vulture]\nunknown_key = True\n'
    var_24 = 'bad_key.toml'
    var_25 = '--config'
    var_26 = 'src'
    var_27 = [var_25, var_3, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--exclude'
    var_30 = 'test.py,temp/*'
    var_31 = '--ignore-names'
    var_32 = 'unused_var'
    var_33 = [var_9, var_29, var_30, var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = b'[tool.vulture]\npaths = ["io_path.py"]'
    var_36 = [var_9]
    var_37 = [var_9]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the merging logic of make_config. \n    Note: Test 2 is designed to fail if paths are empty, so we handle it via pytest.raises.\n    '
    var_1 = None
    var_2 = 'utf-8'
    var_3 = str(var_2)
    var_4 = 'dummy_path'

def test_case_0():
    var_0 = 'Test that providing a wrong type via CLI raises InputError.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'path/to/code'
    var_3 = [var_2]
    var_4 = str(var_2)

def test_case_0():
    var_0 = 'Test that an unknown key in TOML raises InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = True'
    var_2 = 'path/to/code'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that the function attempts to read pyproject.toml from disk if not provided.'
    var_1 = 'some_path'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)



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
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = 'my_dir'
    var_10 = '20'
    var_11 = [var_9, var_3, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n[tool.vulture]\nmin_confidence = 10\nexclude = ["test_*.py"]\nmake_whitelist = true\npaths = ["from_toml"]\n'
    var_14 = 'pyproject.toml'
    var_15 = 'cli_path'
    var_16 = '80'
    var_17 = [var_15, var_3, var_16]
    var_18 = 'cli_path'
    var_19 = [var_18]
    var_20 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_21 = 'path'
    var_22 = [var_21]
    var_23 = b'\n[tool.vulture]\nunknown_key = True\npaths = ["p"]\n'
    var_24 = 'path'
    var_25 = [var_24]
    var_26 = 'path'
    var_27 = '--exclude'
    var_28 = 'a.py,b.py'
    var_29 = '--ignore-names'
    var_30 = 'func1,func2'
    var_31 = [var_26, var_27, var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = [var_26]
    var_34 = module_0.make_config(var_33)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = ''
    var_1 = b'dummy content'

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '10'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = 'tool'
    var_1 = 'vulture'
    var_2 = 'invalid_key'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = b'dummy content'
    var_8 = 'path/to/code'
    var_9 = [var_8]

import vulture.config as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = '--exclude'
    var_1 = 'file1.py,file2.py'
    var_2 = 'path/to/code'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'vulture'
    var_1 = 'src/'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nmin_confidence = 80\nexclude = ["test/*"]\nverbose = true\n'
    var_7 = 'pyproject.toml'
    var_8 = 'vulture'
    var_9 = 'my_dir'
    var_10 = [var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = '20'
    var_13 = [var_8, var_9, var_10, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = 'Namespace'
    var_16 = ()
    var_17 = 'paths'
    var_18 = 'min_confidence'
    var_19 = 'config'
    var_20 = '.'
    var_21 = [var_20]
    var_22 = 'not_an_int'
    var_23 = 'p.toml'
    var_24 = {var_17: var_21, var_18: var_22, var_19: var_23}
    var_25 = 'vulture'
    var_26 = [var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = str(var_22)
    var_29 = 'Namespace'
    var_30 = ()
    var_31 = 'paths'
    var_32 = 'config'
    var_33 = []
    var_34 = 'p.toml'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = []
    var_37 = module_0.make_config(var_36)
    var_38 = str(var_24)
    var_39 = '--exclude'
    var_40 = 'a,b,c'
    var_41 = '--ignore-names'
    var_42 = 'name1,name2'
    var_43 = [var_29, var_30, var_39, var_40, var_41, var_42]
    var_44 = module_0.make_config(var_43)



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test*.py"]\nverbose = true\n'
    var_1 = 'pyproject.toml'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--paths'
    var_5 = 'src/'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '.'
    var_9 = [var_4, var_8]
    var_10 = '--min-confidence'
    var_11 = 'not_an_int'
    var_12 = [var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--paths'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '--paths'
    var_18 = 'folder'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'some_path'
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--sort-by-size'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test*.py"]\nverbose = true\n'
    var_7 = 'pyproject.toml'
    var_8 = 'src/'
    var_9 = [var_8, var_1, var_2]
    var_10 = module_0.make_config(var_9)
    var_11 = b'\n[tool.vulture]\nmin_confidence = "high"\n'
    var_12 = 'path/'
    var_13 = [var_12]
    var_14 = module_0.make_config(var_13, var_2)
    var_15 = str(var_12)
    var_16 = b'\n[tool.vulture]\nunknown_key = True\n'
    var_17 = 'path/'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18, var_2)
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = str(var_20)
    var_23 = 'path/'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/'
    var_1 = '--exclude'
    var_2 = 'a.py,b.py'
    var_3 = '--ignore-names'
    var_4 = 'name1,name2'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)



# Parsed testcases at query #12
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
    var_9 = 'test_dir'
    var_10 = [var_9]
    var_11 = module_0.object()
    var_12 = module_0.object()
    var_13 = module_0.object()
    var_14 = module_0.object()
    var_15 = module_0.object()
    var_16 = module_0.object()
    var_17 = module_0.object()
    var_18 = 'test_dir'
    var_19 = [var_18]
    var_20 = module_1.make_config(var_19)
    var_21 = b'[tool.vulture]\nmin_confidence = 50\nverbose = false\npaths = ["toml_path"]'
    var_22 = 'pyproject.toml'
    var_23 = '--min-confidence'
    var_24 = '75'
    var_25 = 'some_path'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_1.make_config(var_26)
    var_28 = 'path'
    var_29 = [var_28]
    var_30 = 'pyproject.toml'
    var_31 = 'not_an_int'
    var_32 = module_0.object()
    var_33 = module_0.object()
    var_34 = module_0.object()
    var_35 = module_0.object()
    var_36 = module_0.object()
    var_37 = module_0.object()
    var_38 = [var_28]
    var_39 = module_1.make_config(var_38)
    var_40 = str(var_28)
    var_41 = []
    var_42 = 'pyproject.toml'
    var_43 = module_0.object()
    var_44 = module_0.object()
    var_45 = module_0.object()
    var_46 = module_0.object()
    var_47 = module_0.object()
    var_48 = module_0.object()
    var_49 = module_0.object()
    var_50 = []
    var_51 = module_1.make_config(var_50)
    var_52 = str(var_41)
    var_53 = b'[tool.vulture]\nmin_confidence = 25\npaths = ["direct_path"]'
    var_54 = 'cli_path'
    var_55 = [var_54]
    var_56 = 'pyproject.toml'
    var_57 = module_0.object()
    var_58 = module_0.object()
    var_59 = module_0.object()
    var_60 = module_0.object()
    var_61 = module_0.object()
    var_62 = module_0.object()
    var_63 = module_0.object()
    var_64 = [var_54]



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'utf-8'

def test_case_0():
    var_0 = 'Test that providing no paths raises an InputError.'
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = [var_1, var_2]
    var_4 = b''

def test_case_0():
    var_0 = 'Test that incorrect types in TOML raise an InputError.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = '.'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'Test that unknown keys in TOML raise an InputError.'
    var_1 = b'[tool.vulture]\nunknown_key = True'
    var_2 = '.'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that the function attempts to load from disk if no tomlfile is provided.'
    var_1 = b'[tool.vulture]\nmin_confidence = 25'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path/to/dir'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--sort-by-size'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = 'my_folder'
    var_9 = '20'
    var_10 = [var_8, var_3, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = b'\n[tool.vulture]\nexclude = ["test*.py"]\nmin_confidence = 10\nverbose = true\n'
    var_13 = 'pyproject.toml'
    var_14 = 'some_path'
    var_15 = '80'
    var_16 = [var_14, var_3, var_15]
    var_17 = module_0.make_config(var_16, var_2)
    var_18 = 'path'
    var_19 = '--min-confidence'
    var_20 = 'not_an_int'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = 'obj'
    var_24 = 'paths'
    var_25 = 'exclude'
    var_26 = 'ignore_decorators'
    var_27 = 'ignore_names'
    var_28 = 'make_whitelist'
    var_29 = 'min_confidence'
    var_30 = 'sort_by_size'
    var_31 = 'config'
    var_32 = 'verbose'
    var_33 = 'unknown_key'
    var_34 = 'p'
    var_35 = [var_34]
    var_36 = []
    var_37 = []
    var_38 = []
    var_39 = False
    var_40 = 'cfg.toml'
    var_41 = 'error'
    var_42 = {var_24: var_35, var_25: var_36, var_26: var_37, var_27: var_38, var_28: var_39, var_29: var_39, var_30: var_39, var_31: var_40, var_32: var_39, var_33: var_41}
    var_43 = 'path'
    var_44 = [var_43]
    var_45 = module_0.make_config(var_44)
    var_46 = 'path'
    var_47 = '--exclude'
    var_48 = 'a.py,b.py'
    var_49 = '--ignore-names'
    var_50 = 'func1,func2'
    var_51 = [var_46, var_47, var_48, var_49, var_50]
    var_52 = module_0.make_config(var_51)
    var_53 = b'[tool.vulture]\nwrong_key = true'
    var_54 = 'path'
    var_55 = [var_54]
    var_56 = b'[tool.vulture]\nmin_confidence = "high"'
    var_57 = 'path'
    var_58 = [var_57]



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = b'dummy content'
    var_1 = b''
    var_2 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing no paths in both CLI and TOML raises InputError.'
    var_1 = 'paths'
    var_2 = []
    var_3 = (var_1, var_2)
    var_4 = ''
    var_5 = [var_4]
    var_6 = module_0.make_config(var_5)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing wrong types (e.g. string for int) raises InputError.'
    var_1 = '.'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)

def test_case_0():
    var_0 = 'Test that an unknown key in TOML raises InputError.'
    var_1 = 'tool'
    var_2 = 'vulture'
    var_3 = 'unknown_key'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = b''
    var_9 = '.'
    var_10 = [var_9]



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 'utf-8'

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing no paths raises InputError.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing wrong type via CLI/TOML raises InputError.'
    var_1 = 'min_confidence'
    var_2 = 'not_an_int'
    var_3 = {var_1: var_2}
    var_4 = module_0._check_input_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that unknown configuration keys raise InputError.'
    var_1 = 'unknown_key'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = module_0._check_input_config(var_3)

def test_case_0():
    var_0 = 'Test the logic where it actually opens a file from disk.'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.vulture]\nmin_confidence = 25'
    var_3 = 'utf-8'
    var_4 = '--config'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/dir'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = 'path1'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = '--sort-by-size'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '--exclude'
    var_11 = 'file1.py,file2.py'
    var_12 = '--ignore-names'
    var_13 = 'name1,name2'
    var_14 = [var_3, var_10, var_11, var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["from_toml.py"]\nverbose = false\n'
    var_17 = 'pyproject.toml'
    var_18 = '--config'
    var_19 = '80'
    var_20 = module_0.make_config(var_14)
    var_21 = b'[tool.vulture]\nmin_confidence = "high"'
    var_22 = 'path1'
    var_23 = [var_22]
    var_24 = b'[tool.vulture]\nunknown_key = true'
    var_25 = 'path1'
    var_26 = [var_25]
    var_27 = []
    var_28 = module_0.make_config(var_27)
    var_29 = str(var_27)
    var_30 = '--make-whitelist'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'utf-8'

import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)

def test_case_0():
    var_0 = '[tool.vulture]\nmin_confidence = "not_an_int"'
    var_1 = 'utf-8'
    var_2 = 'path/to/dir'
    var_3 = [var_2]

def test_case_0():
    var_0 = '[tool.vulture]\nunknown_key = True'
    var_1 = 'utf-8'
    var_2 = 'path/to/dir'
    var_3 = [var_2]

import vulture.config as module_0

def test_case_0():
    var_0 = '[tool.vulture]\nverbose = true'
    var_1 = 'utf-8'
    var_2 = 'path/to/dir'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)

import vulture.config as module_0

def test_case_0():
    var_0 = 'path/to/dir'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'utf-8'
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
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = False
    var_16 = 'pyproject.toml'
    var_17 = {var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_15, var_8: var_15, var_9: var_16, var_10: var_15}
    var_18 = '--min-confidence'
    var_19 = 1
    var_20 = 'path/to/dir'
    var_21 = 'path/to/file.py'

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that providing no paths raises an InputError.'
    var_1 = 'paths'
    var_2 = 'exclude'
    var_3 = 'ignore_decorators'
    var_4 = 'ignore_names'
    var_5 = 'make_whitelist'
    var_6 = 'min_confidence'
    var_7 = 'sort_by_size'
    var_8 = 'config'
    var_9 = 'verbose'
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = False
    var_15 = 'pyproject.toml'
    var_16 = {var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_14, var_7: var_14, var_8: var_15, var_9: var_14}
    var_17 = []
    var_18 = module_0.make_config(var_17)

def test_case_0():
    var_0 = 'Test that invalid types in TOML raise InputError.'
    var_1 = '[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'utf-8'
    var_3 = 'paths'
    var_4 = 'exclude'
    var_5 = 'ignore_decorators'
    var_6 = 'ignore_names'
    var_7 = 'make_whitelist'
    var_8 = 'min_confidence'
    var_9 = 'sort_by_size'
    var_10 = 'config'
    var_11 = 'verbose'
    var_12 = 'test.py'
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = False
    var_18 = 'pyproject.toml'
    var_19 = {var_3: var_13, var_4: var_14, var_5: var_15, var_6: var_16, var_7: var_17, var_8: var_17, var_9: var_17, var_10: var_18, var_11: var_17}
    var_20 = 'test.py'
    var_21 = [var_20]



# Parsed testcases at query #4
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
    var_9 = 'my_dir'
    var_10 = [var_9, var_3, var_4, var_5]
    var_11 = module_0.make_config(var_10)
    var_12 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test*.py"]\npaths = ["from_toml"]\n'
    var_13 = 'pyproject.toml'
    var_14 = 'some_path'
    var_15 = '--config'

import vulture.config as module_0

def test_case_0():
    var_0 = '\n    Comprehensive test for make_config covering:\n    - CLI precedence over TOML\n    - TOML loading\n    - Default values application\n    - Input validation (InputError)\n    '
    var_1 = 'tool'
    var_2 = 'vulture'
    var_3 = 'min_confidence'
    var_4 = 'exclude'
    var_5 = 'paths'
    var_6 = 10
    var_7 = '*.tmp'
    var_8 = [var_7]
    var_9 = 'toml_path'
    var_10 = [var_9]
    var_11 = {var_3: var_6, var_4: var_8, var_5: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = '[[tool.vulture]]\nmin_confidence = 10\nexclude = ["*.tmp"]\npaths = ["toml_path"]'
    var_15 = '[tool.vulture]\nmin_confidence = 10\nexclude = ["*.tmp"]\npaths = ["toml_path"]'
    var_16 = 'pyproject.toml'
    var_17 = 'utf-8'
    var_18 = 'cli_path'
    var_19 = '--min-confidence'
    var_20 = '50'
    var_21 = [var_18, var_19, var_20]
    var_22 = 'path'
    var_23 = '--min-confidence'
    var_24 = 'not_an_int'
    var_25 = [var_22, var_23, var_24]
    var_26 = b'[tool.vulture]\nunknown_key = 123\npaths = ["p"]'
    var_27 = 'path'
    var_28 = [var_27]
    var_29 = b'[tool.vulture]\nmin_confidence = 10'
    var_30 = []
    var_31 = 'path'
    var_32 = [var_31]
    var_33 = b'[tool.vulture]\npaths=["p"]'
    var_34 = '--exclude'
    var_35 = 'a.py,b.py'
    var_36 = '--ignore-names'
    var_37 = 'name1,name2'
    var_38 = [var_31, var_34, var_35, var_36, var_37]
    var_39 = module_0.make_config(var_38)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the merging logic of make_config using both simulated CLI args \n    and mocked TOML file content.\n    '
    var_1 = 'utf-8'
    var_2 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that InputError is raised when no paths are provided in the final config.'
    var_1 = '--config'
    var_2 = 'nonexistent.toml'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = str(var_2)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that InputError is raised when a CLI argument has an incorrect type.'
    var_1 = '--min-confidence'
    var_2 = 'not_an_int'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)

def test_case_0():
    var_0 = 'Tests that InputError is raised when an unknown key is found in TOML.'
    var_1 = '[tool.vulture]\nunknown_key = "value"'
    var_2 = 'utf-8'
    var_3 = 'path/to/dir'
    var_4 = [var_3]



# Parsed testcases at query #6
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = '50'
    var_2 = '--sort-by-size'
    var_3 = 'path/to/dir'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nexclude = ["test*.py"]\nmin_confidence = 20\npaths = ["toml_path"]\n'
    var_7 = 'pyproject.toml'
    var_8 = '80'
    var_9 = 'cli_path'
    var_10 = [var_0, var_8, var_9]
    var_11 = [var_0, var_8, var_9]
    var_12 = '--min-confidence'
    var_13 = 'not_an_int'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'utf-8'

def test_case_0():
    var_0 = 'Test that it raises InputError if no paths are provided.'
    var_1 = '--config'
    var_2 = 'nonexistent.toml'
    var_3 = [var_1, var_2]
    var_4 = b''

def test_case_0():
    var_0 = 'Test that it raises InputError if a type mismatch occurs.'
    var_1 = b'[tool.vulture]\nmin_confidence = "high"'
    var_2 = '--paths'
    var_3 = '.'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'Test that it raises InputError for unknown configuration keys.'
    var_1 = b'[tool.vulture]\nunknown_key = true'
    var_2 = '--paths'
    var_3 = '.'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'Test that comma-separated strings in CLI are parsed into lists.'
    var_1 = '--paths'
    var_2 = '.'
    var_3 = '--exclude'
    var_4 = 'a,b,c'
    var_5 = '--ignore-names'
    var_6 = 'x,y'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = b''



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--sort-by-size'
    var_5 = 'path/to/dir'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'[tool.vulture]\nmin_confidence = 10\nverbose = true\npaths = ["toml_path"]'
    var_9 = 'pyproject.toml'
    var_10 = '20'
    var_11 = 'extra_path'
    var_12 = [var_2, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = b'[tool.vulture]\nexclude = ["*.pyc", "venv"]\npaths = ["."] '
    var_15 = 'config.toml'
    var_16 = '--config'
    var_17 = '--min-confidence'
    var_18 = 'not_an_int'
    var_19 = 'path'
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = b'[tool.vulture]\nunknown_key = True\npaths = ["."]'
    var_23 = 'invalid.toml'
    var_24 = '--config'
    var_25 = [var_24, var_18]
    var_26 = module_0.make_config(var_25)
    var_27 = '--exclude'
    var_28 = 'a,b,c'
    var_29 = 'path'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = b'[tool.vulture]\nmin_confidence = 99\npaths = ["io_path"]'
    var_33 = [var_25]
    var_34 = b'[tool.vulture]\nmin_confidence = 10'
    var_35 = 'no_paths.toml'
    var_36 = '--config'
    var_37 = [var_36, var_18]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #9
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
    var_8 = '--exclude'
    var_9 = 'test.py,venv/*'
    var_10 = '--ignore-names'
    var_11 = 'foo,bar'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n[tool.vulture]\nmin_confidence = 20\nexclude = ["old.py"]\nverbose = false\n'
    var_15 = 'pyproject.toml'
    var_16 = '--min-confidence'
    var_17 = '80'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--min-confidence'
    var_21 = 'not_an_int'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = str(var_22)
    var_25 = b'[tool.vulture]\nunknown_key = true'
    var_26 = []
    var_27 = module_0.make_config(var_26)
    var_28 = str(var_26)
    var_29 = []
    var_30 = module_0.make_config(var_29)
    var_31 = str(var_29)
    var_32 = b'[tool.vulture]\nsort_by_size = true'
    var_33 = 'some_path'
    var_34 = [var_33]



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = None

import vulture.config as module_0

def test_case_0():
    var_0 = '--min-confidence'
    var_1 = 'not_an_int'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = str(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = 'unknown_key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0._check_input_config(var_2)
    var_4 = str(var_0)

def test_case_0():
    var_0 = b'[tool.vulture]\npaths = []'
    var_1 = []
    var_2 = str(var_0)

import vulture.config as module_0

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)

import vulture.config as module_0

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 42\npaths = ["src"]'
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the merging logic of make_config. \n    Since _check_output_config requires 'paths', we must ensure paths is present.\n    "
    var_1 = 'path/to/code'
    var_2 = 'dummy_path'
    var_3 = 'utf-8'
    var_4 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that an error is raised when no paths are provided.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

def test_case_0():
    var_0 = 'Tests that invalid types in TOML raise an InputError.'
    var_1 = '[tool.vulture]\nmin_confidence = "high"'
    var_2 = 'utf-8'
    var_3 = 'path/to/dir'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Tests that unknown keys in TOML raise an InputError.'
    var_1 = '[tool.vulture]\nunknown_key = True'
    var_2 = 'utf-8'
    var_3 = 'path/to/dir'
    var_4 = [var_3]

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests loading from an actual file path using mocks.'
    var_1 = '[tool.vulture]\nmin_confidence = 20'
    var_2 = 'tool'
    var_3 = 'vulture'
    var_4 = 'min_confidence'
    var_5 = 20
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'some_path'
    var_9 = [var_8]
    var_10 = module_0.make_config(var_9)

import vulture.config as module_0

def test_case_0():
    var_0 = 'Tests that CSV arguments are correctly split into lists.'
    var_1 = '--exclude'
    var_2 = 'a.py,b.py'
    var_3 = '--ignore-names'
    var_4 = 'foo,bar'
    var_5 = 'my_dir'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'[tool.vulture]\nmin_confidence = 50\npaths = ["src"]\nverbose = true'
    var_1 = 'pyproject.toml'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'test_file.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5, var_1)
    var_7 = '--exclude'
    var_8 = 'temp.py'
    var_9 = '--sort-by-size'
    var_10 = 'dir/'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '--min-confidence'
    var_14 = 'not_an_int'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = b'[tool.vulture]\nunknown_key = true'
    var_18 = []
    var_19 = module_0.make_config(var_18, var_14)
    var_20 = b'[tool.vulture]\npaths = []'
    var_21 = []
    var_22 = module_0.make_config(var_21, var_14)
    var_23 = 'a.py,b.py'
    var_24 = '--ignore-names'
    var_25 = 'func1,func2'
    var_26 = [var_7, var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = 'path/to/code'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29)



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = module_0.make_config()
    var_2 = module_0.make_config()
    var_3 = b'\n[tool.vulture]\nmin_confidence = 25\nexclude = ["test_*.py"]\nverbose = true\n'
    var_4 = 'pyproject.toml'
    var_5 = module_0.make_config()
    var_6 = module_0.make_config()
    var_7 = module_0.make_config()
    var_8 = str(var_7)
    var_9 = b'\n[tool.vulture]\nmin_confidence = "not-an-int"\n'
    var_10 = 'bad_config.toml'
    var_11 = module_0.make_config()
    var_12 = b'\n[tool.vulture]\nmake_whitelist = true\n'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'tool'
    var_1 = 'vulture'
    var_2 = 'min_confidence'
    var_3 = 'verbose'
    var_4 = 'sort_by_size'
    var_5 = 'paths'
    var_6 = '20'
    var_7 = ''
    var_8 = 20
    var_9 = 0
    var_10 = 'false'
    var_11 = False
    var_12 = False
    var_13 = 'true'
    var_14 = True
    var_15 = False
    var_16 = 'paths = ["src"]'
    var_17 = 'src'
    var_18 = [var_17]
    var_19 = []
    var_20 = 'utf-8'
    var_21 = None

import vulture.config as module_0

def test_case_0():
    var_0 = 'Test that an error is raised if no paths are provided.'
    var_1 = []
    var_2 = module_0.make_config(var_1)

def test_case_0():
    var_0 = 'Test that an error is raised when a type mismatch occurs.'
    var_1 = '[tool.vulture]\nmin_confidence = "not_an_int"'
    var_2 = 'utf-8'
    var_3 = 'path/to/dir'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test that an error is raised when an unknown key is in the config.'
    var_1 = '[tool.vulture]\nunknown_key = true'
    var_2 = 'utf-8'
    var_3 = 'path/to/dir'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test the logic of loading from an actual file on disk.'
    var_1 = 'pyproject.toml'
    var_2 = '[tool.vulture]\nmin_confidence = 75\npaths = ["."] '
    var_3 = '--config'
    var_4 = '.'



# Parsed testcases at query #15
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
    var_9 = 'vulture'
    var_10 = 'some_path.py'
    var_11 = [var_9, var_10]
    var_12 = ''
    var_13 = module_1.make_config(var_11, var_3)
    var_14 = '\n[tool.vulture]\nmin_confidence = 50\nexclude = ["test/*"]\n'
    var_15 = 'path.py'
    var_16 = '--min-confidence'
    var_17 = '80'
    var_18 = [var_9, var_15, var_16, var_17]
    var_19 = '--exclude'
    var_20 = 'a.py,b.py'
    var_21 = '--ignore-names'
    var_22 = 'func1,func2'
    var_23 = [var_9, var_15, var_19, var_20, var_21, var_22]
    var_24 = '--make-whitelist'
    var_25 = '--sort-by-size'
    var_26 = '-v'
    var_27 = [var_9, var_15, var_24, var_25, var_26]
    var_28 = 'not_an_int'
    var_29 = [var_9, var_15, var_16, var_28]
    var_30 = ''
    var_31 = module_1.make_config(var_29, var_10)
    var_32 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_33 = 'vulture'
    var_34 = 'path.py'
    var_35 = [var_33, var_34]
    var_36 = 'vulture'
    var_37 = [var_36]
    var_38 = ''
    var_39 = module_1.make_config(var_37, var_3)
    var_40 = 'pyproject.toml'
    var_41 = '[tool.vulture]\nmin_confidence = 25\n'
    var_42 = 'utf-8'
    var_43 = '--config'



