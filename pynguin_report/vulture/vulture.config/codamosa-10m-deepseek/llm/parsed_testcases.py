####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = '--make-whitelist'
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = '--exclude'
    var_10 = 'test_*.py,venv'
    var_11 = '--ignore-decorators'
    var_12 = '@app.route'
    var_13 = '--ignore-names'
    var_14 = 'visit_*'
    var_15 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_0]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 30\n    verbose = true\n    '
    var_18 = 0
    var_19 = '80'
    var_20 = [var_7, var_19]
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_24 = '\n    [tool.vulture]\n    verbose = "string instead of bool"\n    '
    var_25 = '--min-confidence'
    var_26 = 'not_an_int'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["test_*.py", "venv"]\n    min_confidence = 50\n    verbose = true\n    paths = ["src"]\n    '
    var_5 = []
    var_6 = '--verbose'
    var_7 = 'mypath.py'
    var_8 = [var_6, var_7]
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = '--help'
    var_12 = [var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--version'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n[tool.vulture]\npaths = ["toml_path1", "toml_path2"]\nmin_confidence = 50\nverbose = true\n'
    var_8 = []
    var_9 = '\n[tool.vulture]\npaths = ["toml_path1"]\nmin_confidence = 50\n'
    var_10 = 'cli_path1'
    var_11 = '--min-confidence'
    var_12 = '80'
    var_13 = [var_10, var_11, var_12]
    var_14 = '--exclude'
    var_15 = 'test_*.py,venv'
    var_16 = [var_2, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = '--ignore-names'
    var_21 = 'visit_*,do_*'
    var_22 = [var_2, var_18, var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--make-whitelist'
    var_25 = '--sort-by-size'
    var_26 = [var_2, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--config'
    var_29 = 'nonexistent.toml'
    var_30 = [var_2, var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["path1"]\n'
    var_33 = []
    var_34 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["path1"]\n'
    var_35 = []
    var_36 = '\n[tool.vulture]\nverbose = true\n'
    var_37 = []



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = 'path.py'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 30\n    verbose = true\n    paths = ["src/"]\n    '
    var_11 = '80'
    var_12 = [var_4, var_11]
    var_13 = b'\n    [tool.vulture]\n    sort_by_size = true\n    make_whitelist = true\n    '
    var_14 = []
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '--exclude'
    var_18 = 'test_*.py,docs'
    var_19 = '--ignore-decorators'
    var_20 = '@app.route,@require_*'
    var_21 = '--ignore-names'
    var_22 = 'visit_*,do_*'
    var_23 = '--make-whitelist'
    var_24 = '75'
    var_25 = '--sort-by-size'
    var_26 = '--config'
    var_27 = 'custom.toml'
    var_28 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_4, var_24, var_25, var_6, var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = b'\n    [tool.vulture]\n    unknown_key = true\n    '
    var_31 = 'path.py'
    var_32 = [var_31]
    var_33 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_34 = 'path.py'
    var_35 = [var_34]
    var_36 = '--min-confidence'
    var_37 = 'not_an_int'
    var_38 = 'path.py'
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with CLI arguments and TOML file.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = 'path.py'
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = b'\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 30\n    verbose = true\n    paths = ["src/"]\n    '
    var_12 = '80'
    var_13 = 'custom.py'
    var_14 = [var_6, var_12, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = 'test.py'
    var_18 = '--invalid-option'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--make-whitelist'
    var_22 = '--sort-by-size'
    var_23 = 'test.py'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--exclude'
    var_27 = 'file1.py,file2.py,dir/'
    var_28 = '--ignore-decorators'
    var_29 = '@app.route,@login_required'
    var_30 = [var_26, var_27, var_28, var_29, var_23]
    var_31 = module_0.make_config(var_30)
    var_32 = b'\n    [tool.vulture]\n    exclude = ["file*.py"]\n    ignore_decorators = ["@deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 90\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_33 = []



# Parsed testcases at query #6
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    min_confidence = 50\n    ignore_names = ["test_*"]\n    verbose = true\n    paths = ["src/"]\n    '
    var_5 = 'extra_path.py'
    var_6 = [var_5]
    var_7 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = false\n    '
    var_8 = '--min-confidence'
    var_9 = '80'
    var_10 = '--verbose'
    var_11 = 'path.py'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = '--exclude'
    var_14 = 'test_*.py,*.pyc'
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@login_required'
    var_17 = '--ignore-names'
    var_18 = 'helper_*,internal_*'
    var_19 = '--make-whitelist'
    var_20 = '--sort-by-size'
    var_21 = '75'
    var_22 = 'file1.py'
    var_23 = 'dir/'
    var_24 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_8, var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--config'
    var_27 = 'custom_config.toml'
    var_28 = [var_26, var_27, var_11]
    var_29 = module_0.make_config(var_28)
    var_30 = b''
    var_31 = [var_11]
    var_32 = b'[tool.vulture]\n'
    var_33 = [var_11]
    var_34 = 'test.py'
    var_35 = [var_34]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = '--exclude'
    var_11 = 'test*.py,venv'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '\n[tool.vulture]\nverbose = true\nmin_confidence = 30\npaths = ["src", "tests"]\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nverbose = false\nmin_confidence = 10\n'
    var_17 = '80'
    var_18 = [var_2, var_8, var_17]
    var_19 = '\n[tool.vulture]\nexclude = ["*.pyc", "__pycache__"]\nignore_decorators = ["@login_required"]\nignore_names = ["test_*"]\n'
    var_20 = []
    var_21 = '\n[tool.vulture]\nverbose = true\n'
    var_22 = '--config'
    var_23 = []
    var_24 = module_0.make_config(var_23)
    var_25 = '\n[tool.vulture]\nunknown_key = true\n'
    var_26 = []
    var_27 = '\n[tool.vulture]\nverbose = "not_a_boolean"\n'
    var_28 = []



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    verbose = true\n    min_confidence = 50\n    paths = ["src/", "tests/"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '--min-confidence'
    var_14 = '80'
    var_15 = [var_13, var_14]
    var_16 = '--exclude'
    var_17 = 'file1.py,file2.py,dir/'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--ignore-decorators'
    var_21 = '@app.route,@require_*'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--ignore-names'
    var_25 = 'visit_*,do_*'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--config'
    var_32 = 'custom_config.toml'
    var_33 = [var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)
    var_37 = '--unknown-key'
    var_38 = 'value'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '--min-confidence'
    var_42 = 'not_an_int'
    var_43 = [var_41, var_42]
    var_44 = module_0.make_config(var_43)
    var_45 = 'path1.py'
    var_46 = 'path2.py'
    var_47 = 'path3/'
    var_48 = [var_45, var_46, var_47]
    var_49 = module_0.make_config(var_48)
    var_50 = '\n    [tool.vulture]\n    sort_by_size = true\n    '
    var_51 = '75'
    var_52 = 'test_*'
    var_53 = 'src/'
    var_54 = [var_44, var_13, var_51, var_16, var_52, var_53]



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["test_*.py", "venv"]\nignore_decorators = ["decorator1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 30\nsort_by_size = true\nverbose = true\npaths = ["src"]\n'
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11, var_2]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '--config'
    var_16 = 'nonexistent.toml'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_11 = []
    var_12 = '80'
    var_13 = [var_6, var_12, var_5]
    var_14 = '--config'
    var_15 = 'nonexistent.toml'
    var_16 = 'test_path'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)
    var_22 = b'[tool.vulture]\nunknown_key = true'
    var_23 = []
    var_24 = '--min-confidence'
    var_25 = 'not_an_int'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26)



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = 'test.py'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nmin_confidence = 50\nverbose = true\npaths = ["src/"]\n'
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '80'
    var_13 = 'other.py'
    var_14 = [var_11, var_12, var_4, var_13]
    var_15 = '[tool.vulture]\n'
    var_16 = 'file.py'
    var_17 = [var_4, var_16]
    var_18 = [var_6]
    var_19 = module_0.make_config(var_18)
    var_20 = '--exclude'
    var_21 = '*.pyc,__pycache__'
    var_22 = [var_20, var_21, var_6]
    var_23 = module_0.make_config(var_22)
    var_24 = '--ignore-decorators'
    var_25 = '@app.route,@login_required'
    var_26 = [var_24, var_25, var_6]
    var_27 = module_0.make_config(var_26)
    var_28 = '--ignore-names'
    var_29 = 'private_*,helper_*'
    var_30 = [var_28, var_29, var_6]
    var_31 = module_0.make_config(var_30)
    var_32 = '[tool.vulture]\nunknown_key = true'
    var_33 = 'test.py'
    var_34 = [var_33]
    var_35 = '[tool.vulture]\nmin_confidence = "not_an_int"'
    var_36 = 'test.py'
    var_37 = [var_36]
    var_38 = []
    var_39 = module_0.make_config(var_38)
    var_40 = '\n[tool.vulture]\nexclude = ["excluded.py"]\nignore_decorators = ["@decorator1"]\nignore_names = ["name1"]\nmake_whitelist = true\nmin_confidence = 75\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_41 = []
    var_42 = '--config'
    var_43 = 'custom.toml'
    var_44 = [var_42, var_43, var_6]
    var_45 = module_0.make_config(var_44)
    var_46 = 'file1.py'
    var_47 = 'file2.py'
    var_48 = 'dir1/'
    var_49 = [var_46, var_47, var_48]
    var_50 = module_0.make_config(var_49)
    var_51 = '--make-whitelist'
    var_52 = [var_51, var_6]
    var_53 = module_0.make_config(var_52)
    var_54 = '100'
    var_55 = [var_11, var_54, var_6]
    var_56 = module_0.make_config(var_55)
    var_57 = "[tool.other]\nkey = 'value'\n[tool.vulture]\n"
    var_58 = [var_6]
    var_59 = "[tool.other]\nkey = 'value'\n"
    var_60 = [var_6]
    var_61 = "[tool.vulture]\nverbose = true\npaths = ['test.py']\n"
    var_62 = []



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = '--sort-by-size'
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11, var_1)
    var_13 = b'\n    [tool.vulture]\n    verbose = true\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_14 = []
    var_15 = '80'
    var_16 = 'path_cli'
    var_17 = [var_8, var_15, var_5, var_16]
    var_18 = []
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 50\npaths = ["toml_path"]\nverbose = true\n'
    var_9 = []
    var_10 = '\n[tool.vulture]\nmin_confidence = 50\npaths = ["toml_path"]\n'
    var_11 = '--min-confidence'
    var_12 = '80'
    var_13 = 'cli_path'
    var_14 = [var_11, var_12, var_13]
    var_15 = '--config'
    var_16 = 'nonexistent.toml'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17, var_1)
    var_19 = '--verbose'
    var_20 = [var_19]
    var_21 = None
    var_22 = module_0.make_config(var_20, var_21)



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["path3", "path4"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = false\n    '
    var_12 = '80'
    var_13 = [var_3, var_12, var_2]
    var_14 = '--exclude'
    var_15 = 'test_*.py,docs'
    var_16 = '--ignore-decorators'
    var_17 = '@app.route,@require_*'
    var_18 = '--ignore-names'
    var_19 = 'visit_*,do_*'
    var_20 = [var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--make-whitelist'
    var_23 = '--sort-by-size'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = [var_2]
    var_27 = module_0.make_config(var_26)
    var_28 = []
    var_29 = module_0.make_config(var_28)
    var_30 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test.py"]\n    '
    var_31 = []



# Parsed testcases at query #15
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'path'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n[tool.vulture]\nmin_confidence = 30\nignore_decorators = ["deco1", "deco2"]\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = b'\n[tool.vulture]\nmin_confidence = 30\npaths = ["toml_path"]\n'
    var_13 = '80'
    var_14 = 'cli_path'
    var_15 = [var_5, var_13, var_14]
    var_16 = b'\n[tool.vulture]\nunknown_key = "value"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = '--min-confidence'
    var_21 = 'not_an_int'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--make-whitelist'
    var_25 = '--sort-by-size'
    var_26 = [var_24, var_25, var_7]
    var_27 = module_0.make_config(var_26)



# Parsed testcases at query #16
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'path1'
    var_8 = 'path2'
    var_9 = [var_4, var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = false\n    paths = ["toml_path"]\n    '
    var_12 = [var_4]
    var_13 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    make_whitelist = true\n    sort_by_size = true\n    '
    var_16 = []
    var_17 = ''
    var_18 = []
    var_19 = '\n    [tool.other]\n    value = 42\n    '
    var_20 = []
    var_21 = '--exclude'
    var_22 = 'file1.py,file2.py'
    var_23 = '--ignore-decorators'
    var_24 = 'dec1,dec2'
    var_25 = '--ignore-names'
    var_26 = 'name1,name2'
    var_27 = [var_21, var_22, var_23, var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = '--sort-by-size'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--verbose'
    var_34 = [var_33]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #17
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various combinations of CLI args and TOML config.'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)
    var_6 = '\n    [tool.vulture]\n    paths = ["src/", "tests/"]\n    exclude = ["*.pyc", "__pycache__"]\n    min_confidence = 80\n    verbose = true\n    sort_by_size = true\n    '
    var_7 = 'utf-8'
    var_8 = []
    var_9 = '\n    [tool.vulture]\n    paths = ["toml_path/"]\n    min_confidence = 50\n    verbose = false\n    '
    var_10 = 'cli_path.py'
    var_11 = '--verbose'
    var_12 = '--min-confidence'
    var_13 = '90'
    var_14 = [var_10, var_11, var_12, var_13]
    var_15 = 'path1.py'
    var_16 = 'path2.py'
    var_17 = '--exclude'
    var_18 = 'test_*,docs'
    var_19 = '--ignore-decorators'
    var_20 = '@staticmethod,@classmethod'
    var_21 = '--ignore-names'
    var_22 = 'private_*,internal_*'
    var_23 = '--make-whitelist'
    var_24 = '--sort-by-size'
    var_25 = '75'
    var_26 = '--config'
    var_27 = 'custom_config.toml'
    var_28 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_11, var_12, var_25, var_26, var_27]
    var_29 = module_0.make_config(var_28, var_4)
    var_30 = []
    var_31 = None
    var_32 = module_0.make_config(var_30, var_31)
    var_33 = '\n    [tool.vulture]\n    paths = ["test.py"]\n    unknown_key = "value"\n    '
    var_34 = []
    var_35 = '\n    [tool.vulture]\n    paths = ["test.py"]\n    verbose = "not_a_bool"\n    '
    var_36 = []
    var_37 = 'test.py'
    var_38 = '--min-confidence'
    var_39 = 'not_an_int'
    var_40 = [var_37, var_38, var_39]
    var_41 = None
    var_42 = module_0.make_config(var_40, var_41)
    var_43 = '\n    [tool.other_tool]\n    some_setting = true\n    '
    var_44 = 'test.py'
    var_45 = [var_44]
    var_46 = '\n    [tool.vulture]\n    paths = ["module1/", "module2/"]\n    exclude = ["old_*"]\n    ignore_decorators = ["@deprecated"]\n    ignore_names = ["unused_*"]\n    make_whitelist = true\n    min_confidence = 100\n    sort_by_size = true\n    verbose = true\n    '
    var_47 = []



# Parsed testcases at query #18
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    '
    var_13 = '80'
    var_14 = [var_6, var_13, var_5]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '--config'
    var_19 = 'custom.toml'
    var_20 = 'test_path'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = 'src'
    var_24 = 'tests'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '\n    [tool.vulture]\n    paths = ["src", "lib"]\n    '
    var_28 = []
    var_29 = '--ignore-names'
    var_30 = 'name1,name2'
    var_31 = '--ignore-decorators'
    var_32 = 'dec1,dec2'
    var_33 = [var_29, var_30, var_31, var_32, var_20]
    var_34 = module_0.make_config(var_33)



# Parsed testcases at query #19
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nverbose = true\nmin_confidence = 20\n'
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = b''
    var_14 = []
    var_15 = b'[tool.vulture]'
    var_16 = []
    var_17 = '--exclude'
    var_18 = 'file*.py,dir/'
    var_19 = '--ignore-decorators'
    var_20 = '@app.route,@require_*'
    var_21 = [var_17, var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--make-whitelist'
    var_24 = '--sort-by-size'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--config'
    var_28 = 'nonexistent.toml'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = 'path1.py'
    var_32 = 'path2.py'
    var_33 = 'dir/'
    var_34 = [var_31, var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = []
    var_37 = module_0.make_config(var_36)
    var_38 = b'[tool.vulture]\nunknown_key = true'
    var_39 = 'path1'
    var_40 = [var_39]
    var_41 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_42 = 'path1'
    var_43 = [var_42]



# Parsed testcases at query #20
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = b'\n    [tool.vulture]\n    paths = ["src/"]\n    '
    var_2 = '--verbose'
    var_3 = 'src/'
    var_4 = [var_2, var_3]
    var_5 = b'[tool.vulture]\nverbose = false'
    var_6 = [var_2, var_3]
    var_7 = module_0.make_config(var_6)
    var_8 = b'[tool.vulture]\nmin_confidence = 50'
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = '--exclude'
    var_12 = '*.pyc,test_*'
    var_13 = [var_11, var_12, var_3]
    var_14 = module_0.make_config(var_13)
    var_15 = '--make-whitelist'
    var_16 = '--sort-by-size'
    var_17 = [var_15, var_16, var_3]
    var_18 = module_0.make_config(var_17)
    var_19 = b'[tool.vulture]\nmin_confidence = 75'
    var_20 = '--config'
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = [var_2, var_3]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #21
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    verbose = true\n    paths = ["src/"]\n    '
    var_9 = 0
    var_10 = '--min-confidence'
    var_11 = '80'
    var_12 = [var_10, var_11]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '--make-whitelist'
    var_16 = 'file.py'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--ignore-decorators'
    var_20 = '@app.route,@require_*'
    var_21 = '--ignore-names'
    var_22 = 'visit_*,do_*'
    var_23 = [var_19, var_20, var_21, var_22, var_16]
    var_24 = module_0.make_config(var_23)
    var_25 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["test.py"]\n    '



# Parsed testcases at query #22
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--make-whitelist'
    var_5 = '--sort-by-size'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@app.route"]\nignore_names = ["visit_*"]\nmin_confidence = 50\nsort_by_size = true\npaths = ["src/"]\nverbose = true\n'
    var_11 = 'utf-8'
    var_12 = []
    var_13 = '--min-confidence=80'
    var_14 = 'new_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = b'[tool.vulture]\ninvalid_key = true'
    var_20 = []
    var_21 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_22 = []



# Parsed testcases at query #23
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    verbose = true\n    min_confidence = 75\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    verbose = false\n    min_confidence = 10\n    '
    var_13 = '90'
    var_14 = [var_3, var_4, var_13]
    var_15 = '--exclude'
    var_16 = 'test_*.py,*.pyc'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--ignore-decorators'
    var_20 = '@app.route,@require_*'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--ignore-names'
    var_24 = 'visit_*,do_*'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--make-whitelist'
    var_28 = '--sort-by-size'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = []
    var_32 = module_0.make_config(var_31)
    var_33 = '--config'
    var_34 = 'nonexistent.toml'
    var_35 = [var_33, var_34]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #24
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    min_confidence = 30\n    verbose = false\n    paths = ["toml_path"]\n    '
    var_12 = '80'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_2, var_13]
    var_15 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    '
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    make_whitelist = true\n    sort_by_size = true\n    '
    var_18 = []
    var_19 = b'\n    [tool.vulture]\n    unknown_key = "test"\n    paths = ["test.py"]\n    '
    var_20 = []
    var_21 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_22 = []
    var_23 = []
    var_24 = module_0.make_config(var_23)
    var_25 = '--exclude'
    var_26 = 'file1.py,file2.py'
    var_27 = '--ignore-decorators'
    var_28 = 'deco1,deco2'
    var_29 = '--ignore-names'
    var_30 = 'name1,name2'
    var_31 = [var_25, var_26, var_27, var_28, var_29, var_30, var_5, var_6]
    var_32 = module_0.make_config(var_31)
    var_33 = b'\n        [tool.vulture]\n        min_confidence = 75\n        paths = ["toml_path"]\n        '
    var_34 = '--config'
    var_35 = 'nonexistent.toml'
    var_36 = 'test_path'
    var_37 = [var_34, var_35, var_36]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #25
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--sort-by-size'
    var_5 = 'path1.py'
    var_6 = 'path2.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 50\n    verbose = true\n    exclude = ["test_*.py"]\n    '
    var_10 = [var_4]
    var_11 = 0
    var_12 = '--min-confidence'
    var_13 = '80'
    var_14 = [var_12, var_13, var_3]
    var_15 = '--exclude'
    var_16 = 'dir1,dir2'
    var_17 = '--ignore-decorators'
    var_18 = 'deco1,deco2'
    var_19 = '--ignore-names'
    var_20 = 'name1,name2'
    var_21 = '--make-whitelist'
    var_22 = '100'
    var_23 = '-v'
    var_24 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_12, var_22, var_4, var_23, var_5, var_6]
    var_25 = module_0.make_config(var_24)
    var_26 = '--verbose'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = b'\n        [tool.vulture]\n        unknown_key = true\n        paths = ["test.py"]\n        '
    var_30 = []



# Parsed testcases at query #26
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_11 = []
    var_12 = '80'
    var_13 = 'path3'
    var_14 = [var_4, var_12, var_3, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n[tool.vulture]\nunknown_key = true\n'
    var_19 = 'path1'
    var_20 = [var_19]
    var_21 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_22 = 'path1'
    var_23 = [var_22]
    var_24 = '\n[tool.vulture]\nverbose = true\npaths = ["test_path"]\n'
    var_25 = '--config'
    var_26 = module_0.make_config(var_23)



# Parsed testcases at query #27
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    paths = ["src/"]\n    min_confidence = 50\n    verbose = true\n    '
    var_5 = 'utf-8'
    var_6 = []
    var_7 = '--min-confidence'
    var_8 = '80'
    var_9 = 'custom_path.py'
    var_10 = [var_7, var_8, var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = b'[tool.vulture]\n'
    var_14 = []
    var_15 = b"[tool.vulture]\nunknown_key = true\npaths = ['test.py']"
    var_16 = []
    var_17 = b"[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test.py']"
    var_18 = []



# Parsed testcases at query #28
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    ignore_decorators = ["deprecated"]\n    verbose = true\n    paths = ["src/"]\n    '
    var_10 = [var_2]
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '\n        [tool.vulture]\n        paths = ["custom_path.py"]\n        '
    var_16 = '--config'
    var_17 = module_0.make_config(var_14)
    var_18 = '--exclude'
    var_19 = 'test_*.py,dir1'
    var_20 = '--ignore-decorators'
    var_21 = '@app.route'
    var_22 = '--ignore-names'
    var_23 = 'helper_*'
    var_24 = '--make-whitelist'
    var_25 = '--sort-by-size'
    var_26 = 'path1.py'
    var_27 = 'path2.py'
    var_28 = [var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25, var_14, var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = 'file.py'
    var_31 = [var_30]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #29
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    make_whitelist = true\n    min_confidence = 10\n    '
    var_9 = []
    var_10 = '--min-confidence'
    var_11 = '50'
    var_12 = [var_10, var_11]
    var_13 = b''
    var_14 = []
    var_15 = '--exclude'
    var_16 = 'test_*.py,docs'
    var_17 = '--ignore-decorators'
    var_18 = '@app.route'
    var_19 = '--ignore-names'
    var_20 = 'visit_*'
    var_21 = '--make-whitelist'
    var_22 = '80'
    var_23 = [var_2, var_3, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_10, var_22, var_5, var_4]
    var_24 = module_0.make_config(var_23)
    var_25 = '--config'
    var_26 = 'custom_config.toml'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = 'file1.py'
    var_30 = 'file2.py'
    var_31 = 'file3.py'
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = 'paths'
    var_35 = var_33[var_34]
    var_36 = len(var_35)
    assert var_36 == 3
    var_37 = '--make-whitelist'
    var_38 = [var_37]
    var_39 = module_0.make_config(var_38)



# Parsed testcases at query #30
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = b'\n[tool.vulture]\npaths = ["src", "tests"]\nverbose = true\nmin_confidence = 50\n'
    var_6 = []
    var_7 = 0
    var_8 = 'custom_path'
    var_9 = '--min-confidence=80'
    var_10 = [var_8, var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = 'test.py'
    var_14 = [var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--exclude'
    var_6 = 'test_*.py,docs'
    var_7 = '--ignore-decorators'
    var_8 = '@app.route'
    var_9 = '--ignore-names'
    var_10 = 'visit_*'
    var_11 = '--make-whitelist'
    var_12 = '--min-confidence'
    var_13 = '50'
    var_14 = '--sort-by-size'
    var_15 = '--verbose'
    var_16 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_1]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_19 = 'path3.py'
    var_20 = [var_19]
    var_21 = []
    var_22 = b"[tool.other]\nkey = 'value'"
    var_23 = [var_1]
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = b"[tool.vulture]\nunknown_key = 'value'"
    var_27 = 'path1.py'
    var_28 = [var_27]
    var_29 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_30 = 'path1.py'
    var_31 = [var_30]



# Parsed testcases at query #32
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence=50'
    var_5 = 'src/'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["decorator1"]\nsort_by_size = true\n    '
    var_9 = []
    var_10 = '\n[tool.vulture]\nverbose = true\nmin_confidence = 30\n    '
    var_11 = '--min-confidence=80'
    var_12 = [var_3, var_11]
    var_13 = []
    var_14 = b'[tool.vulture]\n'
    var_15 = module_0.make_config(var_13, var_3)
    var_16 = []
    var_17 = b'[tool.vulture]\ninvalid_key = "value"'
    var_18 = module_0.make_config(var_16, var_3)
    var_19 = []
    var_20 = b'[tool.vulture]\nverbose = "yes"'
    var_21 = module_0.make_config(var_19, var_3)



# Parsed testcases at query #33
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--sort-by-size'
    var_8 = '--make-whitelist'
    var_9 = '--exclude'
    var_10 = 'test_*.py,*.pyc'
    var_11 = '--ignore-decorators'
    var_12 = '@app.route,@require_*'
    var_13 = '--ignore-names'
    var_14 = 'visit_*,do_*'
    var_15 = 'path1'
    var_16 = 'path2'
    var_17 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17, var_2)
    var_19 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_20 = 'utf-8'
    var_21 = []
    var_22 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\npaths = ["toml_path1", "toml_path2"]\n'
    var_23 = '80'
    var_24 = 'cli_path1'
    var_25 = [var_5, var_23, var_4, var_24]
    var_26 = '--verbose'
    var_27 = [var_26]
    var_28 = None
    var_29 = module_0.make_config(var_27, var_28)
    var_30 = '--config'
    var_31 = 'custom.toml'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32, var_28)
    var_34 = '\n[other_tool]\nsome_option = true\n'
    var_35 = [var_15]
    var_36 = [var_15]
    var_37 = module_0.make_config(var_36, var_28)



# Parsed testcases at query #34
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various scenarios.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '\n    [tool.vulture]\n    min_confidence = 50\n    verbose = true\n    paths = ["toml_path.py"]\n    '
    var_6 = 'utf-8'
    var_7 = 'cli_path.py'
    var_8 = [var_7]
    var_9 = 0
    var_10 = '--min-confidence=80'
    var_11 = [var_10, var_7]
    var_12 = b'[tool.vulture]\n'
    var_13 = 'test.py'
    var_14 = [var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = b'\n    [tool.vulture]\n    unknown_key = 123\n    '
    var_18 = 'test.py'
    var_19 = [var_18]
    var_20 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_21 = 'test.py'
    var_22 = [var_21]
    var_23 = '--make-whitelist'
    var_24 = [var_23, var_13]
    var_25 = module_0.make_config(var_24)
    var_26 = '--sort-by-size'
    var_27 = [var_26, var_13]
    var_28 = module_0.make_config(var_27)
    var_29 = '--exclude=*.pyc,__pycache__'
    var_30 = [var_29, var_13]
    var_31 = module_0.make_config(var_30)
    var_32 = '--ignore-decorators=@app.route,@login_required'
    var_33 = [var_32, var_13]
    var_34 = module_0.make_config(var_33)
    var_35 = '--ignore-names=visit_*,do_*'
    var_36 = [var_35, var_13]
    var_37 = module_0.make_config(var_36)
    var_38 = '--config=nonexistent.toml'
    var_39 = [var_38, var_13]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #35
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n[tool.vulture]\npaths = ["src/"]\nmin_confidence = 50\nverbose = true\n'
    var_5 = '\n[tool.vulture]\npaths = ["src/"]\nmin_confidence = 50\n'
    var_6 = '--min-confidence'
    var_7 = '80'
    var_8 = 'custom_path.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'test.py'
    var_11 = [var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '--min-confidence'
    var_14 = 'not_an_int'
    var_15 = 'test.py'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["test.py"]\n'
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #36
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["path_from_toml"]\n    verbose = true\n    '
    var_11 = []
    var_12 = 0
    var_13 = '80'
    var_14 = [var_4, var_13]
    var_15 = '--make-whitelist'
    var_16 = 'test.py'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--sort-by-size'
    var_20 = [var_19, var_16]
    var_21 = module_0.make_config(var_20)
    var_22 = '--exclude'
    var_23 = 'test_*.py,*.pyc'
    var_24 = 'src'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--ignore-decorators'
    var_28 = '@app.route,@require_*'
    var_29 = '--ignore-names'
    var_30 = 'visit_*,do_*'
    var_31 = [var_27, var_28, var_29, var_30, var_24]
    var_32 = module_0.make_config(var_31)
    var_33 = []
    var_34 = module_0.make_config(var_33)
    var_35 = '\n    [tool.vulture]\n    unknown_key = "test"\n    paths = ["test.py"]\n    '
    var_36 = []
    var_37 = '--config'
    var_38 = 'nonexistent.toml'
    var_39 = [var_37, var_38, var_16]
    var_40 = module_0.make_config(var_39)
    var_41 = '--version'
    var_42 = [var_41]
    var_43 = module_0.make_config(var_42)



# Parsed testcases at query #37
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = b'\n    [tool.vulture]\n    min_confidence = 50\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_5 = '--min-confidence'
    var_6 = '75'
    var_7 = 'cli_path'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--verbose'
    var_10 = '--sort-by-size'
    var_11 = 'path'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--exclude'
    var_15 = '*.pyc,__pycache__,test_*'
    var_16 = [var_14, var_15, var_11]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = '--ignore-names'
    var_21 = 'visit_*,do_*'
    var_22 = [var_18, var_19, var_20, var_21, var_11]
    var_23 = module_0.make_config(var_22)
    var_24 = '--make-whitelist'
    var_25 = [var_24, var_11]
    var_26 = module_0.make_config(var_25)
    var_27 = [var_11]
    var_28 = module_0.make_config(var_27)
    var_29 = '--config'
    var_30 = 'custom.toml'
    var_31 = [var_29, var_30, var_11]
    var_32 = module_0.make_config(var_31)
    var_33 = [var_11]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)
    var_37 = b'\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test"]\n    '
    var_38 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test"]\n    '



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 50\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_8 = []
    var_9 = '--min-confidence'
    var_10 = '75'
    var_11 = [var_9, var_10, var_2]
    var_12 = b'\n    [tool.vulture]\n    paths = ["test.py"]\n    '
    var_13 = []
    var_14 = '--verbose'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_18 = []
    var_19 = b'\n    [tool.vulture]\n    verbose = "not_a_boolean"\n    paths = ["test.py"]\n    '
    var_20 = []
    var_21 = '--min-confidence'
    var_22 = 'not_an_int'
    var_23 = 'test.py'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)



# Parsed testcases at query #39
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'path.py'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\npaths = ["toml_path.py"]\nmin_confidence = 30\nverbose = true\n'
    var_11 = []
    var_12 = '80'
    var_13 = 'cli_path.py'
    var_14 = [var_5, var_12, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n[tool.vulture]\nunknown_key = true\npaths = ["test.py"]\n'
    var_18 = []
    var_19 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["test.py"]\n'
    var_20 = []
    var_21 = '--exclude'
    var_22 = 'test_*.py,dir'
    var_23 = [var_21, var_22, var_7]
    var_24 = module_0.make_config(var_23)
    var_25 = '--ignore-decorators'
    var_26 = '@app.route,@require_*'
    var_27 = [var_25, var_26, var_7]
    var_28 = module_0.make_config(var_27)
    var_29 = '--ignore-names'
    var_30 = 'visit_*,do_*'
    var_31 = [var_29, var_30, var_7]
    var_32 = module_0.make_config(var_31)
    var_33 = '--make-whitelist'
    var_34 = [var_33, var_7]
    var_35 = module_0.make_config(var_34)
    var_36 = '--sort-by-size'
    var_37 = [var_36, var_7]
    var_38 = module_0.make_config(var_37)
    var_39 = '--config'
    var_40 = 'custom.toml'
    var_41 = [var_39, var_40, var_7]
    var_42 = module_0.make_config(var_41)



# Parsed testcases at query #40
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--sort-by-size'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = [var_3, var_11, var_12]
    var_14 = '--help'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '--version'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = []
    var_21 = b'[tool.vulture]\nmin_confidence = 10'
    var_22 = module_0.make_config(var_20, var_19)
    var_23 = []
    var_24 = b'[tool.vulture]\nunknown_key = true'
    var_25 = module_0.make_config(var_23, var_19)
    var_26 = []
    var_27 = b"[tool.vulture]\npaths = 'single_path'"
    var_28 = module_0.make_config(var_26, var_19)
    var_29 = '--config'
    var_30 = 'nonexistent.toml'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #41
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = '--make-whitelist'
    var_9 = '--sort-by-size'
    var_10 = '--exclude'
    var_11 = 'test_*.py,doc'
    var_12 = '--ignore-decorators'
    var_13 = '@app.route,@require_*'
    var_14 = '--ignore-names'
    var_15 = 'visit_*,do_*'
    var_16 = [var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_1]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_19 = '80'
    var_20 = 'path3'
    var_21 = [var_5, var_19, var_7, var_20]
    var_22 = []
    var_23 = module_0.make_config(var_22)
    var_24 = '[tool.vulture]\ninvalid_key = true'
    var_25 = '[tool.vulture]\nverbose = 123'



# Parsed testcases at query #42
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_10 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\npaths = ["toml_path"]\n'
    var_11 = '90'
    var_12 = [var_3, var_11, var_2]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '\n[tool.vulture]\n'
    var_16 = [var_5, var_6]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\npaths = ["custom_path"]\nverbose = true\n'
    var_19 = '--config'
    var_20 = '\n[tool.vulture]\npaths = ["toml_path1", "toml_path2"]\nverbose = true\n'
    var_21 = 'cli_path'
    var_22 = [var_21]
    var_23 = '\n[tool.vulture]\npaths = ["custom_path"]\n'



# Parsed testcases at query #43
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various scenarios.'
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--verbose'
    var_6 = '--sort-by-size'
    var_7 = 'path'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\npaths = ["toml_path1", "toml_path2"]\nverbose = true\nmin_confidence = 50\n'
    var_11 = []
    var_12 = '--min-confidence'
    var_13 = '80'
    var_14 = 'cli_path'
    var_15 = [var_12, var_13, var_14]
    var_16 = '--verbose'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = [var_7]
    var_20 = ''



# Parsed testcases at query #44
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = '--exclude'
    var_7 = 'test*,docs'
    var_8 = '--ignore-decorators'
    var_9 = '@app.route,@require_*'
    var_10 = '--ignore-names'
    var_11 = 'visit_*,do_*'
    var_12 = '--make-whitelist'
    var_13 = '--min-confidence'
    var_14 = '50'
    var_15 = '--sort-by-size'
    var_16 = '--verbose'
    var_17 = '--config'
    var_18 = 'custom_config.toml'
    var_19 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = module_0.make_config(var_19, var_2)
    var_21 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["toml_exclude*"]\n    ignore_decorators = ["toml_decorator"]\n    ignore_names = ["toml_name"]\n    make_whitelist = true\n    min_confidence = 30\n    sort_by_size = true\n    verbose = true\n    '
    var_22 = []
    var_23 = '80'
    var_24 = 'cli_path'
    var_25 = [var_13, var_23, var_16, var_24]
    var_26 = '\n    [tool.vulture]\n    min_confidence = 20\n    verbose = true\n    '
    var_27 = []
    var_28 = []
    var_29 = ''
    var_30 = module_0.make_config(var_28, var_2)
    var_31 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_32 = []
    var_33 = module_0.make_config(var_32, var_29)
    var_34 = '\n    [tool.vulture]\n    min_confidence = "not_an_integer"\n    '
    var_35 = []
    var_36 = module_0.make_config(var_35, var_29)



# Parsed testcases at query #45
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1"]\n    exclude = ["test_*.py"]\n    '
    var_11 = '80'
    var_12 = [var_4, var_11]
    var_13 = '\n    [tool.vulture]\n    sort_by_size = true\n    make_whitelist = true\n    ignore_decorators = ["decorator1"]\n    ignore_names = ["name1", "name2"]\n    '
    var_14 = []
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_18 = 'path1'
    var_19 = [var_18]
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_integer"\n    '
    var_21 = 'path1'
    var_22 = [var_21]
    var_23 = '--exclude'
    var_24 = 'test_*.py,docs'
    var_25 = '--ignore-decorators'
    var_26 = '@app.route,@require_*'
    var_27 = '--ignore-names'
    var_28 = 'visit_*,do_*'
    var_29 = [var_23, var_24, var_25, var_26, var_27, var_28, var_6, var_7]
    var_30 = module_0.make_config(var_29)
    var_31 = '--make-whitelist'
    var_32 = '--sort-by-size'
    var_33 = [var_31, var_32, var_3, var_6]
    var_34 = module_0.make_config(var_33)
    var_35 = '--config'
    var_36 = 'custom_config.toml'
    var_37 = [var_35, var_36, var_6]
    var_38 = module_0.make_config(var_37)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["test_*.py", "build/"]\n    ignore_decorators = ["decorator1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 30\n    sort_by_size = true\n    verbose = false\n    paths = ["src/", "tests/"]\n    '
    var_11 = []
    var_12 = '80'
    var_13 = [var_5, var_6, var_12]
    var_14 = '--config'
    var_15 = 'custom.toml'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--exclude'
    var_19 = 'file1.py,file2.py'
    var_20 = '--ignore-decorators'
    var_21 = 'dec1,dec2'
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '--verbose'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '\n    [tool.vulture]\n    nonexistent_key = true\n    '
    var_30 = 'path1'
    var_31 = [var_30]
    var_32 = 'path1'
    var_33 = '--min-confidence'
    var_34 = 'not_an_int'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = 'custom_path.py'
    var_38 = [var_37]



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various scenarios including TOML and CLI args.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--verbose'
    var_6 = '--sort-by-size'
    var_7 = 'path.py'
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    sort_by_size = true\n    paths = ["path1.py", "path2.py"]\n    '
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    paths = ["path1.py"]\n    '
    var_12 = '--min-confidence'
    var_13 = '50'
    var_14 = [var_12, var_13, var_5, var_2]
    var_15 = '\n    [tool.vulture]\n    '
    var_16 = [var_5, var_7]
    var_17 = '\n        [tool.vulture]\n        exclude = ["test_*.py"]\n        paths = ["path1.py"]\n        '
    var_18 = '--config'
    var_19 = module_0.make_config(var_1)
    var_20 = [var_7]
    var_21 = module_0.make_config(var_20)
    var_22 = []
    var_23 = module_0.make_config(var_22)
    var_24 = '--exclude'
    var_25 = 'test_*.py,venv'
    var_26 = '--ignore-decorators'
    var_27 = '@app.route,@require_*'
    var_28 = '--ignore-names'
    var_29 = 'visit_*,do_*'
    var_30 = '--make-whitelist'
    var_31 = '75'
    var_32 = [var_24, var_25, var_26, var_27, var_28, var_29, var_30, var_6, var_5, var_12, var_31, var_23, var_2]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence=50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_2)
    var_10 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nmake_whitelist = true\nmin_confidence = 10\nverbose = true\npaths = ["path1", "path2"]\n'
    var_11 = []
    var_12 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\npaths = ["path1", "path2"]\n'
    var_13 = 'path3'
    var_14 = [var_5, var_4, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["path1"]\n'
    var_19 = []
    var_20 = '\n[tool.vulture]\nverbose = "not_bool"\npaths = ["path1"]\n'
    var_21 = []
    var_22 = '--sort-by-size'
    var_23 = '--make-whitelist'
    var_24 = [var_22, var_23, var_6]
    var_25 = module_0.make_config(var_24, var_17)
    var_26 = '--exclude=*.pyc,__pycache__'
    var_27 = '--ignore-decorators=@app.route,@login_required'
    var_28 = '--ignore-names=helper_*,utility_*'
    var_29 = [var_26, var_27, var_28, var_6]
    var_30 = module_0.make_config(var_29, var_17)
    var_31 = '--config=nonexistent.toml'
    var_32 = [var_31, var_6]
    var_33 = module_0.make_config(var_32, var_17)



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = b'\n[tool.vulture]\npaths = ["path1", "path2"]\nmin_confidence = 50\nverbose = false\n'
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '75'
    var_7 = [var_4, var_5, var_6]
    var_8 = 'path/to/file.py'
    var_9 = [var_8, var_4]
    var_10 = module_0.make_config(var_9, var_1)
    var_11 = '--exclude'
    var_12 = 'test_*,docs'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13, var_1)
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@require_*'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17, var_1)
    var_19 = '--ignore-names'
    var_20 = 'visit_*,do_*'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21, var_1)
    var_23 = '--make-whitelist'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24, var_1)
    var_26 = '--sort-by-size'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = b'\n[tool.vulture]\nverbose = true\nmin_confidence = 30\n'
    var_30 = '--config'
    var_31 = []
    var_32 = b'[tool.vulture]\n'
    var_33 = module_0.make_config(var_31, var_4)
    var_34 = []
    var_35 = b'[tool.vulture]\nunknown_key = true'
    var_36 = module_0.make_config(var_34, var_4)
    var_37 = '--min-confidence'
    var_38 = 'not_an_int'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'test.py'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    paths = ["toml_path.py"]\n    verbose = true\n    min_confidence = 30\n    '
    var_11 = 'another_path.py'
    var_12 = [var_11]
    var_13 = []
    var_14 = '--make-whitelist'
    var_15 = '--sort-by-size'
    var_16 = [var_14, var_15, var_7]
    var_17 = module_0.make_config(var_16)
    var_18 = '--exclude'
    var_19 = '*.pyc,__pycache__'
    var_20 = '--ignore-decorators'
    var_21 = '@staticmethod,@classmethod'
    var_22 = '--ignore-names'
    var_23 = 'helper_*,internal_*'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_7]
    var_25 = module_0.make_config(var_24)
    var_26 = '--unknown-option'
    var_27 = 'test.py'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--min-confidence'
    var_31 = 'not_an_int'
    var_32 = 'test.py'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #6
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = '\n    [tool.vulture]\n    paths = ["test.py"]\n    min_confidence = 50\n    '
    var_2 = []
    var_3 = '--min-confidence'
    var_4 = '75'
    var_5 = '--verbose'
    var_6 = 'test_cli.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = 'path1.py'
    var_9 = 'path2.py'
    var_10 = 'path3.py'
    var_11 = [var_8, var_9, var_10]
    var_12 = ''
    var_13 = '--exclude'
    var_14 = '*.pyc,__pycache__,*.egg'
    var_15 = '.'
    var_16 = [var_13, var_14, var_15]
    var_17 = '--ignore-decorators'
    var_18 = '@app.route,@require_*'
    var_19 = '--ignore-names'
    var_20 = 'visit_*,do_*'
    var_21 = [var_17, var_18, var_19, var_20, var_15]
    var_22 = '--make-whitelist'
    var_23 = '--sort-by-size'
    var_24 = [var_22, var_23, var_15]
    var_25 = '--config'
    var_26 = 'custom_config.toml'
    var_27 = [var_25, var_26, var_15]
    var_28 = []
    var_29 = ''
    var_30 = module_0.make_config(var_28, var_3)
    var_31 = '--unknown-option'
    var_32 = '.'
    var_33 = [var_31, var_32]
    var_34 = '[tool.vulture]\nunknown_key = true'
    var_35 = module_0.make_config(var_33, var_5)
    var_36 = '--min-confidence'
    var_37 = 'not_an_int'
    var_38 = '.'
    var_39 = [var_36, var_37, var_38]
    var_40 = ''
    var_41 = module_0.make_config(var_39, var_35)



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'path.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = '--sort-by-size'
    var_9 = '--make-whitelist'
    var_10 = '--exclude'
    var_11 = 'test*,docs'
    var_12 = '--ignore-decorators'
    var_13 = '@app.route'
    var_14 = '--ignore-names'
    var_15 = 'helper_*'
    var_16 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\npaths = ["toml_path1.py", "toml_path2.py"]\nmin_confidence = 30\nverbose = true\nexclude = ["venv", "__pycache__"]\n'
    var_19 = 'utf-8'
    var_20 = []
    var_21 = 0
    var_22 = 'cli_path.py'
    var_23 = '80'
    var_24 = [var_22, var_5, var_23]
    var_25 = []
    var_26 = module_0.make_config(var_25)
    var_27 = '--version'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--help'
    var_31 = [var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--config'
    var_34 = 'nonexistent.toml'
    var_35 = [var_4, var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = '\n[tool.vulture]\npaths = ["p1.py", "p2.py"]\nmin_confidence = 100\nverbose = true\nsort_by_size = true\nmake_whitelist = true\nexclude = ["ex1", "ex2"]\nignore_decorators = ["dec1", "dec2"]\nignore_names = ["name1", "name2"]\n'
    var_38 = []



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["test_path.py"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_3, var_10]
    var_12 = '--exclude'
    var_13 = 'test_*.py,docs'
    var_14 = '--ignore-decorators'
    var_15 = '@app.route,@require_*'
    var_16 = '--ignore-names'
    var_17 = 'visit_*,do_*'
    var_18 = '--make-whitelist'
    var_19 = '--sort-by-size'
    var_20 = '--config'
    var_21 = 'custom_config.toml'
    var_22 = 'path2.py'
    var_23 = [var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_5, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = b'\n    [tool.vulture]\n    invalid_key = true\n    '
    var_26 = []
    var_27 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_28 = []
    var_29 = '--make-whitelist'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30)



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_2)
    var_9 = b'\n[tool.vulture]\npaths = ["test_path"]\nverbose = true\nmin_confidence = 50\n'
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '80'
    var_13 = [var_11, var_12]
    var_14 = '--exclude'
    var_15 = 'test_*.py,venv'
    var_16 = 'src'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17, var_2)
    var_19 = '--ignore-decorators'
    var_20 = '@app.route'
    var_21 = '--ignore-names'
    var_22 = 'visit_*'
    var_23 = [var_19, var_20, var_21, var_22, var_16]
    var_24 = module_0.make_config(var_23, var_2)
    var_25 = '--make-whitelist'
    var_26 = '--sort-by-size'
    var_27 = [var_25, var_26, var_16]
    var_28 = module_0.make_config(var_27, var_2)
    var_29 = '--version'
    var_30 = [var_29]
    var_31 = None
    var_32 = module_0.make_config(var_30, var_31)
    var_33 = []
    var_34 = None
    var_35 = module_0.make_config(var_33, var_34)
    var_36 = b'\n[tool.vulture]\ninvalid_key = true\npaths = ["test"]\n'
    var_37 = []
    var_38 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["test"]\n'
    var_39 = []



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--make-whitelist'
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@app.route"]\nmin_confidence = 50\nverbose = true\npaths = ["src/"]\n'
    var_10 = 'file.py'
    var_11 = [var_10]
    var_12 = '\n[tool.vulture]\nverbose = true\nmin_confidence = 80\n'
    var_13 = '--min-confidence'
    var_14 = '30'
    var_15 = 'path.py'
    var_16 = [var_3, var_13, var_14, var_15]
    var_17 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_18 = 'test.py'
    var_19 = [var_18]
    var_20 = '\n[tool.vulture]\nverbose = "not_a_bool"\n'
    var_21 = 'test.py'
    var_22 = [var_21]
    var_23 = []
    var_24 = None
    var_25 = module_0.make_config(var_23, var_24)
    var_26 = '  '
    var_27 = 'test.py'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28, var_24)
    var_30 = '--sort-by-size'
    var_31 = [var_30, var_27]
    var_32 = module_0.make_config(var_31, var_24)
    var_33 = '--exclude'
    var_34 = 'file1.py,file2.py'
    var_35 = '--ignore-names'
    var_36 = 'name1,name2'
    var_37 = [var_33, var_34, var_35, var_36, var_27]
    var_38 = module_0.make_config(var_37, var_24)
    var_39 = '\n[tool.vulture]\nmin_confidence = 90\npaths = ["temp_test.py"]\n'
    var_40 = '--config'



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = 'file.py'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\npaths = ["toml_path.py"]\nmin_confidence = 50\nverbose = true\n'
    var_10 = '--min-confidence'
    var_11 = '80'
    var_12 = [var_6, var_10, var_11]
    var_13 = []
    var_14 = '--exclude'
    var_15 = '*.pyc,__pycache__'
    var_16 = '--ignore-decorators'
    var_17 = '@app.route,@require_*'
    var_18 = '--ignore-names'
    var_19 = 'visit_*,do_*'
    var_20 = [var_14, var_15, var_16, var_17, var_18, var_19, var_6]
    var_21 = module_0.make_config(var_20)
    var_22 = '--make-whitelist'
    var_23 = [var_22, var_6]
    var_24 = module_0.make_config(var_23)
    var_25 = []
    var_26 = module_0.make_config(var_25)
    var_27 = []
    var_28 = '[tool.vulture]\ninvalid_key = true'
    var_29 = module_0.make_config(var_27, var_2)
    var_30 = []
    var_31 = "[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_32 = module_0.make_config(var_30, var_2)
    var_33 = '--config'
    var_34 = 'nonexistent.toml'
    var_35 = [var_33, var_34, var_6]
    var_36 = module_0.make_config(var_35)
    var_37 = [var_32]
    var_38 = "[tool.vulture]\npaths = ['file.py']"
    var_39 = 'test.py'
    var_40 = [var_39]
    var_41 = module_0.make_config(var_40)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 50\n    exclude = ["test_*.py", "docs"]\n    verbose = true\n    paths = ["src"]\n    '
    var_9 = 0
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = false\n    '
    var_12 = '--min-confidence'
    var_13 = '80'
    var_14 = [var_12, var_13, var_2]
    var_15 = '\n        [tool.vulture]\n        min_confidence = 90\n        paths = ["custom_path"]\n        '
    var_16 = '--config'
    var_17 = 'extra_path'
    var_18 = []
    var_19 = b'[tool.vulture]\nmin_confidence = 5'
    var_20 = module_0.make_config(var_18, var_3)
    var_21 = []
    var_22 = b'[tool.vulture]\nunknown_key = true'
    var_23 = module_0.make_config(var_21, var_3)
    var_24 = '--exclude'
    var_25 = 'a.py,b.py'
    var_26 = '--ignore-decorators'
    var_27 = '@dec1,@dec2'
    var_28 = '--ignore-names'
    var_29 = 'name1,name2'
    var_30 = 'test_path'
    var_31 = [var_24, var_25, var_26, var_27, var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--make-whitelist'
    var_34 = 'some_path'
    var_35 = [var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = '42'
    var_38 = [var_12, var_37, var_30]
    var_39 = module_0.make_config(var_38)



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = '--sort-by-size'
    var_7 = 'path1'
    var_8 = 'path2'
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9, var_1)
    var_11 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_12 = []
    var_13 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = false\n'
    var_14 = '80'
    var_15 = [var_3, var_14, var_5]
    var_16 = '--exclude'
    var_17 = 'dir1,dir2'
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@login_required'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20, var_1)
    var_22 = '--ignore-names'
    var_23 = 'helper_,internal_'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24, var_1)
    var_26 = '--make-whitelist'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = '--config'
    var_30 = 'custom_config.toml'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31, var_1)
    var_33 = []
    var_34 = None
    var_35 = module_0.make_config(var_33, var_34)



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various combinations of CLI args and TOML file.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = b'\n    [tool.vulture]\n    paths = ["toml_path.py"]\n    min_confidence = 50\n    verbose = true\n    '
    var_6 = '--min-confidence'
    var_7 = '80'
    var_8 = 'cli_path.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = b'\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    ignore_decorators = ["@app.route"]\n    make_whitelist = true\n    '
    var_11 = 'some_path.py'
    var_12 = [var_11]
    var_13 = b'[tool.vulture]\n'
    var_14 = '--verbose'
    var_15 = '--sort-by-size'
    var_16 = 'path.py'
    var_17 = [var_14, var_15, var_16]
    var_18 = '--exclude'
    var_19 = 'dir/,test_*.py'
    var_20 = '--ignore-names'
    var_21 = 'helper*,utils'
    var_22 = '--make-whitelist'
    var_23 = 'main.py'
    var_24 = 'utils.py'
    var_25 = [var_18, var_19, var_20, var_21, var_22, var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = []
    var_28 = module_0.make_config(var_27)
    var_29 = b'\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test.py"]\n    '
    var_30 = []
    var_31 = module_0.make_config(var_30, var_28)
    var_32 = '--min-confidence'
    var_33 = 'not_an_int'
    var_34 = 'path.py'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = b'\n    [tool.vulture]\n    paths = ["path1.py", "path2.py"]\n    exclude = ["venv", "*.pyc"]\n    ignore_decorators = ["@staticmethod", "@classmethod"]\n    ignore_names = ["__init__"]\n    make_whitelist = false\n    min_confidence = 30\n    sort_by_size = true\n    verbose = false\n    '
    var_38 = b'\n    [tool.vulture]\n    paths = ["toml_only.py"]\n    min_confidence = 10\n    verbose = true\n    exclude = ["test_*"]\n    '
    var_39 = '90'
    var_40 = 'cli_only.py'
    var_41 = [var_36, var_39, var_40]
    var_42 = '--config'
    var_43 = 'custom.toml'
    var_44 = 'test.py'
    var_45 = [var_42, var_43, var_44]
    var_46 = b'[tool.vulture]\nmin_confidence = 50'
    var_47 = 'a,b,c'
    var_48 = '--ignore-decorators'
    var_49 = 'd,e,f'
    var_50 = [var_18, var_47, var_48, var_49, var_16]
    var_51 = module_0.make_config(var_50)



# Parsed testcases at query #15
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'path.py'
    var_5 = '--verbose'
    var_6 = '--sort-by-size'
    var_7 = '--make-whitelist'
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = '--exclude'
    var_11 = 'test_*,docs'
    var_12 = '--ignore-decorators'
    var_13 = '@app.route'
    var_14 = '--ignore-names'
    var_15 = 'helper_*'
    var_16 = '--config'
    var_17 = 'custom.toml'
    var_18 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_21 = 'override.py'
    var_22 = '90'
    var_23 = [var_21, var_8, var_22]
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = 'path.py'
    var_27 = '--invalid-arg'
    var_28 = [var_26, var_27]
    var_29 = b''
    var_30 = module_0.make_config(var_28, var_5)
    var_31 = 'path.py'
    var_32 = '--min-confidence'
    var_33 = 'not_a_number'
    var_34 = [var_31, var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = b'\n    [tool.vulture]\n    min_confidence = "not_a_number"\n    paths = ["path.py"]\n    '
    var_37 = module_0.make_config(tomlfile=var_31)



# Parsed testcases at query #16
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--sort-by-size'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = 'path1.py'
    var_10 = 'path2.py'
    var_11 = [var_9, var_10]
    var_12 = module_0.make_config(var_11, var_1)
    var_13 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_14 = 'utf-8'
    var_15 = []
    var_16 = '90'
    var_17 = [var_5, var_16, var_3]
    var_18 = []
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)
    var_21 = 'path.py'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22, var_19)
    var_24 = '\n[tool.vulture]\nmin_confidence = 20\n'
    var_25 = '--config'
    var_26 = 'test.py'



# Parsed testcases at query #17
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["src"]\n    verbose = true\n    '
    var_1 = []
    var_2 = 0
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_3, var_4, var_5]
    var_7 = 'path1'
    var_8 = 'path2'
    var_9 = [var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = '--exclude'
    var_12 = 'test_*.py,*.pyc'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '--ignore-decorators'
    var_18 = '@app.route,@require_*'
    var_19 = 'test.py'
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--ignore-names'
    var_23 = 'visit_*,do_*'
    var_24 = [var_22, var_23, var_19]
    var_25 = module_0.make_config(var_24)
    var_26 = '--make-whitelist'
    var_27 = [var_26, var_19]
    var_28 = module_0.make_config(var_27)
    var_29 = '--sort-by-size'
    var_30 = [var_29, var_19]
    var_31 = module_0.make_config(var_30)
    var_32 = [var_3, var_19]
    var_33 = module_0.make_config(var_32)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test"]\n    '
    var_38 = []
    var_39 = '\n    [tool.vulture]\n    verbose = "yes"\n    paths = ["test"]\n    '
    var_40 = []



# Parsed testcases at query #18
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various inputs.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = 'path.py'
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = '\n[tool.vulture]\npaths = ["toml_path.py"]\nmin_confidence = 30\nverbose = true\n'
    var_12 = '\n[tool.vulture]\npaths = ["toml_path.py"]\nmin_confidence = 30\n'
    var_13 = '80'
    var_14 = 'cli_path.py'
    var_15 = [var_6, var_13, var_14]
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["path.py"]\n'
    var_19 = '--min-confidence'
    var_20 = 'not_an_int'
    var_21 = 'path.py'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--sort-by-size'
    var_25 = '--make-whitelist'
    var_26 = [var_24, var_25, var_8]
    var_27 = module_0.make_config(var_26)
    var_28 = '--exclude'
    var_29 = 'file1.py,file2.py'
    var_30 = [var_28, var_29, var_8]
    var_31 = module_0.make_config(var_30)
    var_32 = '--ignore-decorators'
    var_33 = '@app.route,@require_*'
    var_34 = '--ignore-names'
    var_35 = 'visit_*,do_*'
    var_36 = [var_32, var_33, var_34, var_35, var_8]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #19
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nverbose = true\nmin_confidence = 30\npaths = ["src/"]\n'
    var_11 = []
    var_12 = '80'
    var_13 = 'other/'
    var_14 = [var_4, var_12, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n[tool.vulture]\ninvalid_key = true\n'
    var_19 = 'path1'
    var_20 = [var_19]
    var_21 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_22 = 'path1'
    var_23 = [var_22]



# Parsed testcases at query #20
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config with various scenarios.'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n[tool.vulture]\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nverbose = true\n'
    var_10 = b'[tool.vulture]\nverbose = true\nmin_confidence = 10\n'
    var_11 = '90'
    var_12 = [var_4, var_11]
    var_13 = 'file1.py'
    var_14 = 'file2.py'
    var_15 = 'dir/'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--exclude'
    var_19 = 'file*.py,dir/,test_*.py'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--ignore-decorators'
    var_23 = '@app.route,@require_*'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--ignore-names'
    var_27 = 'visit_*,do_*'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--make-whitelist'
    var_31 = '--sort-by-size'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--config'
    var_35 = 'custom_config.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36)
    var_38 = []
    var_39 = module_0.make_config(var_38)
    var_40 = b'[tool.vulture]\nunknown_key = true\n'
    var_41 = b'[tool.vulture]\nverbose = 123\n'
    var_42 = '--make-whitelist'
    var_43 = 'not_bool'
    var_44 = [var_42, var_43]
    var_45 = module_0.make_config(var_44)
    var_46 = b'[tool.vulture]\n'
    var_47 = b'[tool.vulture]\nmin_confidence = 75\n'
    var_48 = 'file.py'
    var_49 = [var_48]
    var_50 = b'[tool.vulture]\npaths = ["src/", "tests/"]\n'
    var_51 = b'[tool.vulture]\nverbose = true\npaths = ["file.py"]\n'



# Parsed testcases at query #21
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n[tool.vulture]\npaths = ["toml_path1.py", "toml_path2.py"]\nmin_confidence = 50\nverbose = true\n'
    var_5 = 'cli_path.py'
    var_6 = [var_5]
    var_7 = 'path.py'
    var_8 = '--verbose'
    var_9 = 'false'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'test.py'
    var_12 = [var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--exclude'
    var_15 = 'test.py,*.txt'
    var_16 = '--ignore-decorators'
    var_17 = '@decorator1,@decorator2'
    var_18 = '--ignore-names'
    var_19 = 'name1,name2'
    var_20 = '--make-whitelist'
    var_21 = '--min-confidence'
    var_22 = '80'
    var_23 = '--sort-by-size'
    var_24 = [var_7, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_8]
    var_25 = module_0.make_config(var_24)
    var_26 = []
    var_27 = module_0.make_config(var_26)



# Parsed testcases at query #22
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--exclude'
    var_6 = 'test*,docs'
    var_7 = '--verbose'
    var_8 = '--make-whitelist'
    var_9 = '--sort-by-size'
    var_10 = '--min-confidence'
    var_11 = '50'
    var_12 = '--ignore-decorators'
    var_13 = 'decor1,decor2'
    var_14 = '--ignore-names'
    var_15 = 'name1,name2'
    var_16 = [var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_19 = []
    var_20 = '80'
    var_21 = [var_10, var_20, var_7]
    var_22 = '\n[tool.vulture]\npaths = ["test_path"]\nmin_confidence = 25\n'
    var_23 = '--config'
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["test"]\n'
    var_27 = []
    var_28 = '\n[tool.vulture]\nmin_confidence = "not_an_integer"\npaths = ["test"]\n'
    var_29 = []



# Parsed testcases at query #23
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--sort-by-size'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 50\n    '
    var_9 = []
    var_10 = 0
    var_11 = '--min-confidence'
    var_12 = '75'
    var_13 = [var_11, var_12]
    var_14 = '--make-whitelist'
    var_15 = [var_14, var_4]
    var_16 = module_0.make_config(var_15)
    var_17 = '--exclude'
    var_18 = 'test_*.py,*.bak'
    var_19 = [var_17, var_18, var_4]
    var_20 = module_0.make_config(var_19)
    var_21 = '--ignore-decorators'
    var_22 = '@app.route,@require_*'
    var_23 = [var_21, var_22, var_4]
    var_24 = module_0.make_config(var_23)
    var_25 = '--ignore-names'
    var_26 = 'visit_*,do_*'
    var_27 = [var_25, var_26, var_4]
    var_28 = module_0.make_config(var_27)
    var_29 = []
    var_30 = module_0.make_config(var_29)
    var_31 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["path1"]\n    '
    var_32 = []



# Parsed testcases at query #24
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--sort-by-size'
    var_6 = 'test_file.py'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = [var_11, var_12, var_4]
    var_14 = 'custom_path.py'
    var_15 = [var_14]
    var_16 = b''
    var_17 = 'test.py'
    var_18 = [var_17]
    var_19 = b'[tool]\nother = true'
    var_20 = [var_17]
    var_21 = b"[tool.vulture]\npaths = ['test.py']"
    var_22 = []
    var_23 = 'path1.py'
    var_24 = 'path2.py'
    var_25 = 'path3.py'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--exclude'
    var_29 = 'file1.py,file2.py,dir/'
    var_30 = [var_28, var_29, var_17]
    var_31 = module_0.make_config(var_30)



# Parsed testcases at query #25
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various configurations.'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\nexclude = ["test_*.py", "docs"]\n'
    var_11 = []
    var_12 = 0
    var_13 = '80'
    var_14 = [var_4, var_13]
    var_15 = '--exclude'
    var_16 = 'file1.py,file2.py,dir1'
    var_17 = '--ignore-decorators'
    var_18 = '@app.route,@require_*'
    var_19 = '--ignore-names'
    var_20 = 'visit_*,do_*'
    var_21 = [var_15, var_16, var_17, var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--make-whitelist'
    var_24 = '--sort-by-size'
    var_25 = [var_23, var_24, var_3]
    var_26 = module_0.make_config(var_25)
    var_27 = '--config'
    var_28 = 'custom.toml'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = 'src/main.py'
    var_32 = 'tests/'
    var_33 = [var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)
    var_37 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_38 = []
    var_39 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_40 = []
    var_41 = b''
    var_42 = 'test_path'
    var_43 = [var_3, var_42]



# Parsed testcases at query #26
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nverbose = true\n'
    var_8 = '\n[tool.vulture]\nverbose = false\nmin_confidence = 50\n'
    var_9 = [var_4]
    var_10 = '--verbose'
    var_11 = [var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '--version'
    var_14 = [var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--help'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_20 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'



# Parsed testcases at query #27
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'path.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = '--exclude'
    var_9 = 'test_*.py,docs'
    var_10 = '--ignore-decorators'
    var_11 = '@deco1,@deco2'
    var_12 = '--ignore-names'
    var_13 = 'helper_,_internal'
    var_14 = '--make-whitelist'
    var_15 = '--sort-by-size'
    var_16 = '--config'
    var_17 = 'custom_config.toml'
    var_18 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_21 = '80'
    var_22 = 'cli_path.py'
    var_23 = [var_5, var_21, var_22]
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = b''
    var_27 = b'[tool.vulture]\nverbose = true'
    var_28 = b'[tool.vulture]\ninvalid_key = 123'



# Parsed testcases at query #28
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '--exclude'
    var_10 = 'file1.py,dir1'
    var_11 = '--ignore-decorators'
    var_12 = 'deco1,deco2'
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '--sort-by-size'
    var_16 = '--make-whitelist'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_20 = []
    var_21 = '80'
    var_22 = 'false'
    var_23 = [var_3, var_21, var_2, var_22]
    var_24 = ''
    var_25 = [var_5]
    var_26 = '--verbose'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '[tool.vulture]\ninvalid_key = true'
    var_30 = []
    var_31 = '[tool.vulture]\nverbose = 123'
    var_32 = []
    var_33 = '--version'
    var_34 = [var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = '--help'
    var_37 = [var_36]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #29
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n[tool.vulture]\npaths = ["src/"]\nexclude = ["test_*.py"]\nverbose = true\n'
    var_11 = []
    var_12 = '\n[tool.vulture]\npaths = ["src/"]\nverbose = false\n'
    var_13 = 'custom_path'
    var_14 = [var_3, var_13]
    var_15 = '--exclude'
    var_16 = '*.pyc,__pycache__'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17, var_1)
    var_19 = '--ignore-decorators'
    var_20 = '@app.route,@require_*'
    var_21 = '--ignore-names'
    var_22 = 'visit_*,do_*'
    var_23 = [var_19, var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23, var_1)
    var_25 = '--make-whitelist'
    var_26 = '--sort-by-size'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = []
    var_30 = None
    var_31 = module_0.make_config(var_29, var_30)
    var_32 = '--unknown-key'
    var_33 = [var_32]
    var_34 = None
    var_35 = module_0.make_config(var_33, var_34)



# Parsed testcases at query #30
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n[tool.vulture]\npaths = ["toml_path.py"]\nmin_confidence = 50\nverbose = true\n'
    var_5 = '--min-confidence'
    var_6 = '80'
    var_7 = 'cli_path.py'
    var_8 = [var_5, var_6, var_7]
    var_9 = []
    var_10 = b'[tool.vulture]\n'
    var_11 = 'test.py'
    var_12 = [var_11]
    var_13 = '--exclude'
    var_14 = 'test_*,venv'
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@login_required'
    var_17 = '--ignore-names'
    var_18 = 'helper_*,temp_*'
    var_19 = '--make-whitelist'
    var_20 = '--sort-by-size'
    var_21 = '--verbose'
    var_22 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_0, var_1]
    var_23 = module_0.make_config(var_22)
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = b'\n[tool.vulture]\ninvalid_key = true\n'
    var_27 = 'test.py'
    var_28 = [var_27]
    var_29 = '--min-confidence'
    var_30 = 'not_an_int'
    var_31 = 'test.py'
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = '--sort-by-size'
    var_8 = '--make-whitelist'
    var_9 = '--exclude'
    var_10 = 'test_*.py,venv'
    var_11 = '--ignore-decorators'
    var_12 = '@app.route,@require_*'
    var_13 = '--ignore-names'
    var_14 = 'visit_*,do_*'
    var_15 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_0, var_1]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_18 = []
    var_19 = '\n[tool.vulture]\nmin_confidence = 10\npaths = ["toml_path"]\n'
    var_20 = '--min-confidence'
    var_21 = '80'
    var_22 = 'cli_path'
    var_23 = [var_20, var_21, var_22]
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '\n[tool.vulture]\nunknown_key = "value"\npaths = ["path1"]\n'
    var_27 = []
    var_28 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["path1"]\n'
    var_29 = []



# Parsed testcases at query #32
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--sort-by-size'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 50\n    exclude = ["file*.py"]\n    verbose = true\n    paths = ["path1"]\n    '
    var_10 = []
    var_11 = 0
    var_12 = '--min-confidence'
    var_13 = '80'
    var_14 = 'custom_path'
    var_15 = [var_12, var_13, var_14]
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test.py"]\n    '
    var_20 = []
    var_21 = '\n    [tool.vulture]\n    verbose = "yes"\n    paths = ["test.py"]\n    '
    var_22 = []



# Parsed testcases at query #33
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = None
    var_3 = module_0.make_config(var_1, var_2)
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'path1'
    var_8 = 'path2'
    var_9 = [var_4, var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9, var_2)
    var_11 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_12 = []
    var_13 = 0
    var_14 = '80'
    var_15 = [var_5, var_14, var_4]
    var_16 = '--version'
    var_17 = [var_16]
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '--help'
    var_21 = [var_20]
    var_22 = None
    var_23 = module_0.make_config(var_21, var_22)
    var_24 = '--make-whitelist'
    var_25 = 'test.py'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_22)
    var_28 = '--ignore-names'
    var_29 = 'visit_*,do_*'
    var_30 = [var_28, var_29, var_25]
    var_31 = module_0.make_config(var_30, var_22)
    var_32 = '--exclude'
    var_33 = '*settings.py,docs,*/test_*.py'
    var_34 = [var_32, var_33, var_25]
    var_35 = module_0.make_config(var_34, var_22)



# Parsed testcases at query #34
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'path.py'
    var_5 = '--exclude=test.py,*.txt'
    var_6 = '--ignore-decorators=decorator1,decorator2'
    var_7 = '--ignore-names=name1,name2'
    var_8 = '--make-whitelist'
    var_9 = '--min-confidence=50'
    var_10 = '--sort-by-size'
    var_11 = '--verbose'
    var_12 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '\n    [tool.vulture]\n    paths = ["src/", "tests/"]\n    exclude = ["*.pyc"]\n    ignore_decorators = ["deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    '
    var_15 = []
    var_16 = 0
    var_17 = 'custom_path.py'
    var_18 = '--min-confidence=80'
    var_19 = [var_17, var_18, var_11]
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = b'[tool.vulture]\npaths = []\n'
    var_23 = []
    var_24 = b"[tool.vulture]\nunknown_key = true\npaths = ['test.py']\n"
    var_25 = []
    var_26 = b"[tool.vulture]\nmin_confidence = 'string'\npaths = ['test.py']\n"
    var_27 = []



# Parsed testcases at query #35
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--verbose'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_6 = []
    var_7 = '--min-confidence=50'
    var_8 = [var_2, var_7]
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = b"[tool.other]\nkey = 'value'"
    var_12 = [var_9]
    var_13 = b"[tool.vulture]\nunknown_key = 'value'"
    var_14 = 'path1'
    var_15 = [var_14]
    var_16 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_17 = 'path1'
    var_18 = [var_17]



# Parsed testcases at query #36
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    min_confidence = 50\n    verbose = true\n    paths = ["toml_path.py"]\n    '
    var_5 = 'cli_path.py'
    var_6 = [var_5]
    var_7 = b"[build-system]\nrequires = ['setuptools']"
    var_8 = 'test.py'
    var_9 = [var_8]
    var_10 = [var_8]
    var_11 = module_0.make_config(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = '\n        [tool.vulture]\n        min_confidence = 80\n        '
    var_15 = '--config'
    var_16 = 'test.py'
    var_17 = module_0.make_config(var_2)



# Parsed testcases at query #37
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    paths = ["src"]\n    exclude = ["test_*.py"]\n    verbose = true\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    paths = ["src"]\n    verbose = false\n    '
    var_13 = 'custom_path'
    var_14 = [var_3, var_13]
    var_15 = '--version'
    var_16 = [var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--help'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--verbose'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--exclude'
    var_25 = 'dir1,dir2'
    var_26 = [var_24, var_25, var_6]
    var_27 = module_0.make_config(var_26)
    var_28 = '--ignore-decorators'
    var_29 = '@app.route,@login_required'
    var_30 = '--ignore-names'
    var_31 = 'visit_*,do_*'
    var_32 = [var_28, var_29, var_30, var_31, var_6]
    var_33 = module_0.make_config(var_32)
    var_34 = '--make-whitelist'
    var_35 = '--sort-by-size'
    var_36 = [var_34, var_35, var_6]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'test_path'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    verbose = true\n    '
    var_10 = []
    var_11 = b'[tool.vulture]\nverbose = true'
    var_12 = '--exclude'
    var_13 = 'test_*.py,dir2'
    var_14 = [var_3, var_12, var_13]
    var_15 = []
    var_16 = b'[tool.vulture]\nmin_confidence = 10'
    var_17 = module_0.make_config(var_15, var_3)
    var_18 = b'[tool.vulture]\npaths = ["src/", "tests/"]'
    var_19 = []
    var_20 = []
    var_21 = b'[tool.vulture]\nunknown_key = true'
    var_22 = module_0.make_config(var_20, var_3)
    var_23 = []
    var_24 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_25 = module_0.make_config(var_23, var_3)
    var_26 = [var_6, var_3]
    var_27 = module_0.make_config(var_26, var_24)



# Parsed testcases at query #39
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = []
    var_2 = module_0.make_config(var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_11 = 0
    var_12 = '80'
    var_13 = [var_4, var_12, var_3]
    var_14 = b''
    var_15 = [var_6]
    var_16 = b"[tool.other]\nkey = 'value'"
    var_17 = [var_6]
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = b"[tool.vulture]\nmin_confidence = 'string'"
    var_21 = b"[tool.vulture]\nunknown_key = 'value'"
    var_22 = '--version'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--help'
    var_26 = [var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--config'
    var_29 = 'custom.toml'
    var_30 = [var_28, var_29, var_6]
    var_31 = module_0.make_config(var_30)
    var_32 = '--make-whitelist'
    var_33 = [var_32, var_6]
    var_34 = module_0.make_config(var_33)
    var_35 = '--sort-by-size'
    var_36 = [var_35, var_6]
    var_37 = module_0.make_config(var_36)
    var_38 = '--ignore-decorators'
    var_39 = '@app.route,@require_*'
    var_40 = '--ignore-names'
    var_41 = 'visit_*,do_*'
    var_42 = [var_38, var_39, var_40, var_41, var_6]
    var_43 = module_0.make_config(var_42)
    var_44 = '--exclude'
    var_45 = '*settings.py,docs,*/test_*.py,venv'
    var_46 = [var_44, var_45, var_6]
    var_47 = module_0.make_config(var_46)



# Parsed testcases at query #40
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--exclude'
    var_5 = 'test_*.py,docs'
    var_6 = '--ignore-decorators'
    var_7 = '@app.route'
    var_8 = '--ignore-names'
    var_9 = 'visit_*'
    var_10 = '--make-whitelist'
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = '--sort-by-size'
    var_14 = '--verbose'
    var_15 = '--config'
    var_16 = 'custom_config.toml'
    var_17 = [var_0, var_1, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_20 = []
    var_21 = '\n[tool.vulture]\nmin_confidence = 10\nverbose = false\npaths = ["toml_path"]\n'
    var_22 = 'cli_path'
    var_23 = '20'
    var_24 = [var_22, var_11, var_23, var_14]
    var_25 = ''
    var_26 = [var_0]
    var_27 = "[tool.other]\nkey = 'value'"
    var_28 = [var_0]
    var_29 = []
    var_30 = module_0.make_config(var_29)
    var_31 = '\n[tool.vulture]\nunknown_key = true\npaths = ["path1"]\n'
    var_32 = []
    var_33 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["path1"]\n'
    var_34 = []



# Parsed testcases at query #41
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\npaths = ["src/"]\nexclude = ["test_*.py"]\nverbose = true\n'
    var_11 = []
    var_12 = [var_3]
    var_13 = '--exclude'
    var_14 = 'file1.py,file2.py'
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@require_*'
    var_17 = [var_13, var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)
    var_22 = '\n[tool.vulture]\nunknown_key = true\n'
    var_23 = []
    var_24 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_25 = []



# Parsed testcases at query #42
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = 'path.py'
    var_5 = '--exclude=test_*.py,docs'
    var_6 = '--ignore-decorators=@app.route'
    var_7 = '--ignore-names=visit_*'
    var_8 = '--make-whitelist'
    var_9 = '--min-confidence=50'
    var_10 = '--sort-by-size'
    var_11 = '--verbose'
    var_12 = '--config=custom.toml'
    var_13 = [var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_16 = 'utf-8'
    var_17 = 'extra_path.py'
    var_18 = [var_17]
    var_19 = '--verbose=False'
    var_20 = [var_19, var_17]
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = b'[tool.vulture]\ninvalid_key = true'
    var_24 = 'test.py'
    var_25 = [var_24]
    var_26 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_27 = 'test.py'
    var_28 = [var_27]



# Parsed testcases at query #43
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = 'dir1/'
    var_4 = '--exclude'
    var_5 = 'test_*.py,venv'
    var_6 = '--ignore-decorators'
    var_7 = '@app.route,@require_*'
    var_8 = '--ignore-names'
    var_9 = 'visit_*,do_*'
    var_10 = '--make-whitelist'
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = '--sort-by-size'
    var_14 = '--verbose'
    var_15 = '--config'
    var_16 = 'custom.toml'
    var_17 = [var_0, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_20 = []
    var_21 = 'custom_path.py'
    var_22 = '80'
    var_23 = [var_21, var_11, var_22]
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = b'[tool.vulture]\nexclude = []'
    var_27 = []
    var_28 = b"[tool.vulture]\nverbose = true\npaths = ['test.py']"
    var_29 = []



# Parsed testcases at query #44
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'Test make_config function with various scenarios.'
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--verbose'
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = '--sort-by-size'
    var_9 = '--make-whitelist'
    var_10 = '--exclude'
    var_11 = 'test_*.py,docs'
    var_12 = '--ignore-decorators'
    var_13 = '@app.route,@require_*'
    var_14 = '--ignore-names'
    var_15 = 'visit_*,do_*'
    var_16 = [var_1, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_19 = []
    var_20 = '80'
    var_21 = [var_6, var_20, var_5]
    var_22 = []
    var_23 = module_0.make_config(var_22)
    var_24 = b'[tool.vulture]\nunknown_key = true'
    var_25 = []
    var_26 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_27 = []
    var_28 = '--config'
    var_29 = 'custom_config.toml'
    var_30 = 'path.py'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = 'dir1'
    var_34 = 'dir2'
    var_35 = 'file1.py'
    var_36 = 'file2.py'
    var_37 = [var_33, var_34, var_35, var_36]
    var_38 = module_0.make_config(var_37)
    var_39 = b'[tool.vulture]\npaths = ["src"]'
    var_40 = [var_5]



# Parsed testcases at query #45
#--------------------------




