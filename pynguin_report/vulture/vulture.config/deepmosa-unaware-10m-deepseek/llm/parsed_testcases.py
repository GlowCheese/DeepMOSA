####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
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
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'new_path'
    var_13 = '80'
    var_14 = [var_12, var_5, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_21 = []
    var_22 = '--config'
    var_23 = None
    var_24 = module_0.make_config(var_16, var_23)
    var_25 = '--config'
    var_26 = 'nonexistent.toml'
    var_27 = [var_25, var_26, var_23]
    var_28 = module_0.make_config(var_27, var_16)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 30\n    exclude = ["test*.py"]\n    '
    var_10 = 'utf-8'
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '80'
    var_14 = [var_12, var_4, var_13]
    var_15 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_18 = []
    var_19 = []
    var_20 = b'[tool.vulture]\n'
    var_21 = module_0.make_config(var_19, var_3)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    exclude = ["test*.py", "temp/"]\n    min_confidence = 50\n    paths = ["src/", "main.py"]\n    verbose = true\n    '
    var_7 = 'extra.py'
    var_8 = [var_7]
    var_9 = '\n    [tool.vulture]\n    min_confidence = 50\n    paths = ["toml_path.py"]\n    '
    var_10 = '--min-confidence'
    var_11 = '80'
    var_12 = 'cli_path.py'
    var_13 = [var_10, var_11, var_12]
    var_14 = '--exclude'
    var_15 = 'test*.py,docs'
    var_16 = '--ignore-decorators'
    var_17 = '@app.route,@require_*'
    var_18 = '--ignore-names'
    var_19 = 'visit_*,do_*'
    var_20 = '--make-whitelist'
    var_21 = '70'
    var_22 = '--sort-by-size'
    var_23 = '--verbose'
    var_24 = '--config'
    var_25 = 'custom.toml'
    var_26 = [var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_10, var_21, var_22, var_23, var_24, var_25, var_2, var_3]
    var_27 = module_0.make_config(var_26)
    var_28 = '\n    [tool.vulture]\n    paths = ["src/"]\n    '
    var_29 = []
    var_30 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    '
    var_31 = []



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = '--verbose'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["src"]\n    exclude = ["test_*.py"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["src"]\n    '
    var_12 = '--min-confidence'
    var_13 = '20'
    var_14 = 'other'
    var_15 = [var_12, var_13, var_14]
    var_16 = '\n        [tool.vulture]\n        min_confidence = 30\n        paths = ["lib"]\n        '
    var_17 = '--config'
    var_18 = module_0.make_config(var_4)
    var_19 = '\n    [tool.vulture]\n    invalid_key = true\n    '
    var_20 = []
    var_21 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_22 = []
    var_23 = '\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_24 = []
    var_25 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["src"]\n    '
    var_26 = []
    var_27 = True
    var_28 = '--exclude'
    var_29 = 'a.py,b.py'
    var_30 = '--ignore-decorators'
    var_31 = 'dec1,dec2'
    var_32 = '--ignore-names'
    var_33 = 'name1,name2'
    var_34 = [var_28, var_29, var_30, var_31, var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = [var_27]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["path_from_toml"]\n    verbose = true\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test"]\n    '
    var_14 = []
    var_15 = module_0.make_config(var_14, var_2)
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = 'test_file.py'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--config'
    var_22 = 'nonexistent.toml'
    var_23 = 'test.py'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = [var_23]
    var_27 = '[tool.vulture]\n'



# Parsed testcases at query #6
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
    var_9 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nmin_confidence = 10\nsort_by_size = true\npaths = ["path1", "path2"]\n'
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = '--exclude'
    var_14 = 'file1.py,file2.py'
    var_15 = 'path'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = [var_18, var_19, var_15]
    var_21 = module_0.make_config(var_20)
    var_22 = '--ignore-names'
    var_23 = 'visit_*,do_*'
    var_24 = [var_22, var_23, var_15]
    var_25 = module_0.make_config(var_24)
    var_26 = '--make-whitelist'
    var_27 = [var_26, var_15]
    var_28 = module_0.make_config(var_27)
    var_29 = '--sort-by-size'
    var_30 = [var_29, var_15]
    var_31 = module_0.make_config(var_30)
    var_32 = []
    var_33 = module_0.make_config(var_32)
    var_34 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_35 = []
    var_36 = '\n[tool.vulture]\nmin_confidence = "high"\n'
    var_37 = []
    var_38 = '--config'
    var_39 = module_0.make_config(var_33)
    var_40 = []
    var_41 = None
    var_42 = module_0.make_config(var_40, var_41)
    var_43 = '\n[tool.vulture]\npaths = ["path"]\n'
    var_44 = [var_33]
    var_45 = '\n[tool.vulture]\npaths = ["toml_path"]\n'
    var_46 = 'cli_path'
    var_47 = [var_46]



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 75\n    paths = ["toml_path1"]\n    exclude = ["test_*.py"]\n    '
    var_10 = []
    var_11 = '90'
    var_12 = [var_2, var_11]
    var_13 = '--exclude'
    var_14 = 'file1.py,file2.py'
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@require_*'
    var_17 = '--ignore-names'
    var_18 = 'visit_*,do_*'
    var_19 = '--make-whitelist'
    var_20 = '--sort-by-size'
    var_21 = '--config'
    var_22 = 'custom_config.toml'
    var_23 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = []
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11, var_4]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["path1"]\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_18 = []
    var_19 = '--version'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--help'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--exclude'
    var_26 = 'file1.py,file2.py'
    var_27 = [var_25, var_26, var_5]
    var_28 = module_0.make_config(var_27)
    var_29 = '--ignore-decorators'
    var_30 = '@app.route,@require_*'
    var_31 = [var_29, var_30, var_5]
    var_32 = module_0.make_config(var_31)
    var_33 = '--ignore-names'
    var_34 = 'visit_*,do_*'
    var_35 = [var_33, var_34, var_5]
    var_36 = module_0.make_config(var_35)
    var_37 = '--make-whitelist'
    var_38 = [var_37, var_5]
    var_39 = module_0.make_config(var_38)
    var_40 = '--sort-by-size'
    var_41 = [var_40, var_5]
    var_42 = module_0.make_config(var_41)
    var_43 = '--config'
    var_44 = 'custom.toml'
    var_45 = [var_43, var_44, var_5]
    var_46 = module_0.make_config(var_45)



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '20'
    var_12 = [var_4, var_11, var_6]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    paths = ["test.py"]\n    '
    var_20 = []
    var_21 = '--version'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '75'
    var_12 = [var_4, var_5, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = []
    var_18 = ''
    var_19 = module_0.make_config(var_17, var_3)
    var_20 = 'test.py'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = "[tool.vulture]\npaths = ['test.py']"
    var_24 = []



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path_from_toml"]\n    exclude = ["exclude1", "exclude2"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path_from_toml"]\n    '
    var_13 = 'cli_path'
    var_14 = '20'
    var_15 = [var_13, var_5, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = module_0.make_config(var_17, var_1)
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_20 = []
    var_21 = module_0.make_config(var_20, var_1)
    var_22 = []
    var_23 = None
    var_24 = module_0.make_config(var_22, var_23)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_9 = '75'
    var_10 = [var_3, var_9]
    var_11 = '[tool.vulture]\ninvalid_key = 1\n'
    var_12 = "[tool.vulture]\nmin_confidence = 'not_an_int'\n"
    var_13 = '--min-confidence'
    var_14 = '50'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = 'test_file.py'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--config'
    var_21 = 'nonexistent.toml'
    var_22 = [var_17, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--version'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28)



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
    var_6 = 'test_file.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test_*.py"]\n'
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = []
    var_14 = None
    var_15 = module_0.make_config(var_13, var_14)
    var_16 = '--make-whitelist'
    var_17 = '--sort-by-size'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_14)



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_7 = []
    var_8 = '--min-confidence'
    var_9 = '20'
    var_10 = 'cli_path'
    var_11 = [var_8, var_9, var_10]
    var_12 = b'[tool.vulture]\ninvalid_key = true\npaths = ["path"]'
    var_13 = []
    var_14 = b'[tool.vulture]\nmin_confidence = "10"\npaths = ["path"]'
    var_15 = []
    var_16 = b'[tool.vulture]\nmin_confidence = 10'
    var_17 = []
    var_18 = '--config'
    var_19 = 'nonexistent.toml'
    var_20 = [var_18, var_19, var_17]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #15
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '90'
    var_13 = [var_5, var_12, var_7]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '--unknown-arg'
    var_17 = [var_16]
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = []
    var_21 = None
    var_22 = module_0.make_config(var_20, var_21)
    var_23 = 'test.py'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24, var_21)
    var_26 = '--exclude'
    var_27 = 'file1.py,file2.py'
    var_28 = [var_23, var_26, var_27]
    var_29 = module_0.make_config(var_28, var_21)
    var_30 = '--ignore-decorators'
    var_31 = 'deco1,deco2'
    var_32 = [var_23, var_30, var_31]
    var_33 = module_0.make_config(var_32, var_21)
    var_34 = '--ignore-names'
    var_35 = 'name1,name2'
    var_36 = [var_23, var_34, var_35]
    var_37 = module_0.make_config(var_36, var_21)



# Parsed testcases at query #16
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = []
    var_6 = b'\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_7 = '--min-confidence'
    var_8 = '20'
    var_9 = 'path'
    var_10 = [var_7, var_8, var_9]
    var_11 = b'\n    [tool.vulture]\n    paths = ["test_path"]\n    '
    var_12 = []
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test"]\n    '
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test"]\n    '
    var_18 = []



# Parsed testcases at query #17
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1.py'
    var_2 = 'path2.py'
    var_3 = [var_1, var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 50\n    paths = ["path1", "path2"]\n    '
    var_6 = 'cli_path.py'
    var_7 = [var_6]
    var_8 = '--min-confidence'
    var_9 = '20'
    var_10 = [var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path.py"]\n    '
    var_13 = '--invalid-option'
    var_14 = [var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--verbose'
    var_17 = [var_16]



# Parsed testcases at query #18
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_1 = []
    var_2 = '\n    [tool.vulture]\n    exclude = ["file*.py"]\n    min_confidence = 10\n    paths = ["path1"]\n    '
    var_3 = '--min-confidence'
    var_4 = '20'
    var_5 = '--verbose'
    var_6 = 'path3'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = '50'
    var_9 = 'test.py'
    var_10 = [var_3, var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = b''
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "10"\n    paths = ["test.py"]\n    '
    var_17 = []
    var_18 = [var_9]
    var_19 = module_0.make_config(var_18)
    var_20 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["test.py"]\n    '
    var_21 = []



# Parsed testcases at query #19
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '80'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '95'
    var_13 = [var_5, var_12, var_7]
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)
    var_17 = '\n    [tool.vulture]\n    invalid_key = true\n    paths = ["path1"]\n    '
    var_18 = []
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_20 = []
    var_21 = '--min-confidence'
    var_22 = 'not_an_int'
    var_23 = [var_21, var_22]
    var_24 = None
    var_25 = module_0.make_config(var_23, var_24)
    var_26 = '--unknown-arg'
    var_27 = [var_26]
    var_28 = None
    var_29 = module_0.make_config(var_27, var_28)
    var_30 = '--help'
    var_31 = [var_30]
    var_32 = None
    var_33 = module_0.make_config(var_31, var_32)
    var_34 = '--version'
    var_35 = [var_34]
    var_36 = None
    var_37 = module_0.make_config(var_35, var_36)
    var_38 = '--config'
    var_39 = 'custom.toml'
    var_40 = [var_38, var_39]
    var_41 = module_0.make_config(var_40, var_35)
    var_42 = 'nonexistent.toml'
    var_43 = [var_38, var_42, var_36]
    var_44 = module_0.make_config(var_43, var_35)



# Parsed testcases at query #20
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '-v'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = '--min-confidence'
    var_14 = '50'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #21
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 75\n    exclude = ["file*.py"]\n    verbose = true\n    '
    var_9 = []
    var_10 = '90'
    var_11 = [var_4, var_10]
    var_12 = []
    var_13 = '--min-confidence'
    var_14 = '50'
    var_15 = [var_13, var_14]
    var_16 = b''
    var_17 = module_0.make_config(var_15, var_4)
    var_18 = '\n    [tool.vulture]\n    paths = "not_a_list"\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19, var_15)
    var_21 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path1"]\n    '
    var_22 = []
    var_23 = module_0.make_config(var_22, var_15)
    var_24 = '--min-confidence'
    var_25 = 'abc'
    var_26 = 'path1'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = [var_26]
    var_30 = module_0.make_config(var_29)
    var_31 = 'pyproject.toml'
    var_32 = '\n            [tool.vulture]\n            paths = ["src"]\n            min_confidence = 80\n            '
    var_33 = '--config'
    var_34 = module_0.make_config(var_26)



# Parsed testcases at query #22
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    verbose = true\n    '
    var_9 = '70'
    var_10 = [var_2, var_9]
    var_11 = []
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_15 = []
    var_16 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_17 = []
    var_18 = '--version'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--verbose'
    var_22 = [var_21]



# Parsed testcases at query #23
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_10 = []
    var_11 = 'cli_path'
    var_12 = '90'
    var_13 = [var_11, var_4, var_12]
    var_14 = '\n[tool.vulture]\nunknown_option = true\n'
    var_15 = []
    var_16 = module_0.make_config(var_15, var_2)
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = module_0.make_config(var_18, var_2)
    var_20 = []
    var_21 = '[tool.vulture]\n'
    var_22 = module_0.make_config(var_20, var_19)
    var_23 = 'pyproject.toml'
    var_24 = '[tool.vulture]\nmin_confidence = 25\npaths = ["test.py"]\n'
    var_25 = []
    var_26 = None
    var_27 = module_0.make_config(var_25, var_26)
    var_28 = '--version'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--help'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #24
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["file*.py"]\n    ignore_decorators = ["deco1"]\n    verbose = true\n    paths = ["toml_path1"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = '--min-confidence'
    var_18 = '10'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '\n        [tool.vulture]\n        min_confidence = 25\n        paths = ["custom_path"]\n        '
    var_22 = '--config'
    var_23 = module_0.make_config(var_18)
    var_24 = '--version'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--help'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--exclude'
    var_31 = 'file1.py,file2.py'
    var_32 = 'path'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = '\n        [tool.vulture]\n        verbose = true\n        '
    var_36 = '--config'
    var_37 = module_0.make_config(var_28)



# Parsed testcases at query #25
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 10\n    sort_by_size = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_13 = 'cli_path'
    var_14 = '90'
    var_15 = [var_13, var_5, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = []
    var_21 = None
    var_22 = module_0.make_config(var_20, var_21)
    var_23 = '\n        [tool.vulture]\n        min_confidence = 30\n        paths = ["test_path"]\n        '
    var_24 = '--config'
    var_25 = None
    var_26 = module_0.make_config(var_21, var_25)
    var_27 = '\n    [tool.vulture]\n    paths = ["test_path"]\n    '
    var_28 = '--verbose'
    var_29 = [var_28]



# Parsed testcases at query #26
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1.py", "path2.py"]\n    verbose = true\n    '
    var_1 = 'min_confidence'
    var_2 = 20
    var_3 = 'paths'
    var_4 = 'test_file.py'
    var_5 = [var_4]
    var_6 = [var_4]
    var_7 = module_0.make_config(var_6)
    var_8 = 'custom_pyproject.toml'
    var_9 = 'config'
    var_10 = '--config'
    var_11 = [var_10, var_8]
    var_12 = module_0.make_config(var_11)
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_16 = 'paths'
    var_17 = 'test.py'
    var_18 = [var_17]
    var_19 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_20 = 'paths'
    var_21 = 'test.py'
    var_22 = [var_21]
    var_23 = b'\n    [tool.vulture]\n    paths = ["test.py"]\n    verbose = true\n    '
    var_24 = b'\n    [tool.vulture]\n    paths = ["test.py"]\n    verbose = false\n    '
    var_25 = 'paths'
    var_26 = 'test.py'
    var_27 = [var_26]
    var_28 = [var_26]
    var_29 = module_0.make_config(var_28)
    var_30 = b'\n    [tool.vulture]\n    paths = ["toml_path.py"]\n    '
    var_31 = 'paths'
    var_32 = 'cli_path.py'
    var_33 = [var_32]



# Parsed testcases at query #27
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["file*.py"]\nignore_decorators = ["deco1"]\nignore_names = ["name1"]\nmake_whitelist = true\nsort_by_size = true\npaths = ["toml_path1"]\n'
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11, var_4]
    var_13 = '--exclude'
    var_14 = 'file1.py,file2.py'
    var_15 = 'path'
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = '--ignore-names'
    var_21 = 'visit_*,do_*'
    var_22 = [var_18, var_19, var_20, var_21, var_15]
    var_23 = module_0.make_config(var_22)
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '\n[tool.vulture]\nmin_confidence = 90\n'
    var_27 = '--config'
    var_28 = 'test_path'
    var_29 = module_0.make_config(var_3)
    var_30 = b'\n[tool.vulture]\ninvalid_key = 5\npaths = ["test"]\n'
    var_31 = []



# Parsed testcases at query #28
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    sort_by_size = true\n    '
    var_11 = []
    var_12 = '80'
    var_13 = [var_5, var_12]
    var_14 = 'path3'
    var_15 = [var_14]
    var_16 = '--config'
    var_17 = 'nonexistent.toml'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = []
    var_21 = None
    var_22 = module_0.make_config(var_20, var_21)



# Parsed testcases at query #29
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = 'test_config.toml'
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '30'
    var_13 = [var_11, var_12]
    var_14 = '--verbose'
    var_15 = [var_14]
    var_16 = []
    var_17 = 'nonexistent.toml'
    var_18 = module_0.make_config(var_16, var_13)
    var_19 = 'pyproject.toml'
    var_20 = '\n    [tool.vulture]\n    min_confidence = 20\n    paths = ["src"]\n    '
    var_21 = '--config'



# Parsed testcases at query #30
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_12 = '20'
    var_13 = [var_4, var_12]
    var_14 = '--min-confidence'
    var_15 = '10'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_19 = 'path'
    var_20 = [var_19]
    var_21 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_22 = 'path'
    var_23 = [var_22]



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '75'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["src"]\n    exclude = ["test*"]\n    min_confidence = 50\n    verbose = true\n    '
    var_10 = []
    var_11 = '90'
    var_12 = [var_4, var_11]
    var_13 = '--config'
    var_14 = module_0.make_config(var_2)
    var_15 = '\n    [tool.vulture]\n    unknown_key = true\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = ''
    var_21 = module_0.make_config(var_19, var_3)



# Parsed testcases at query #32
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    verbose = true\n    '
    var_9 = '90'
    var_10 = [var_2, var_9]
    var_11 = []
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n    [tool.vulture]\n    invalid_key = true\n    paths = ["path"]\n    '
    var_15 = []
    var_16 = module_0.make_config(var_15, var_13)
    var_17 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_18 = []
    var_19 = module_0.make_config(var_18, var_13)
    var_20 = '--min-confidence'
    var_21 = 'not_an_int'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)



# Parsed testcases at query #33
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1.py'
    var_5 = 'path2.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_15 = []
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    paths = ["path1.py", "path2.py"]\n    '
    var_18 = []
    var_19 = 'test.py'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--exclude'
    var_23 = 'file1.py,file2.py'
    var_24 = [var_22, var_23, var_19]
    var_25 = module_0.make_config(var_24)
    var_26 = '--verbose'
    var_27 = [var_26, var_19]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = [var_29, var_19]
    var_31 = module_0.make_config(var_30)
    var_32 = '--sort-by-size'
    var_33 = [var_32, var_19]
    var_34 = module_0.make_config(var_33)



# Parsed testcases at query #34
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["src"]\n    '
    var_8 = '90'
    var_9 = [var_3, var_8]
    var_10 = '\n    [tool.vulture]\n    invalid_key = true\n    '
    var_11 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_12 = module_0.make_config()
    var_13 = []
    var_14 = module_0.make_config(var_13)



# Parsed testcases at query #35
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = '-v'
    var_13 = [var_2, var_11, var_12]
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = 'pyproject.toml'
    var_17 = '--config'
    var_18 = module_0.make_config(var_3)
    var_19 = b'\n    [tool.vulture]\n    invalid_key = true\n    '
    var_20 = []



# Parsed testcases at query #36
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'test.py'
    var_4 = '--verbose'
    var_5 = [var_3, var_4]
    var_6 = module_0.make_config(var_5, var_1)
    var_7 = 'tool'
    var_8 = 'vulture'
    var_9 = 'exclude'
    var_10 = 'ignore_decorators'
    var_11 = 'ignore_names'
    var_12 = 'make_whitelist'
    var_13 = 'min_confidence'
    var_14 = 'sort_by_size'
    var_15 = 'verbose'
    var_16 = 'paths'
    var_17 = 'file*.py'
    var_18 = 'dir/'
    var_19 = [var_17, var_18]
    var_20 = 'deco1'
    var_21 = 'deco2'
    var_22 = [var_20, var_21]
    var_23 = 'name1'
    var_24 = 'name2'
    var_25 = [var_23, var_24]
    var_26 = True
    var_27 = 10
    var_28 = 'path1'
    var_29 = 'path2'
    var_30 = [var_28, var_29]
    var_31 = {var_9: var_19, var_10: var_22, var_11: var_25, var_12: var_26, var_13: var_27, var_14: var_26, var_15: var_26, var_16: var_30}
    var_32 = {var_8: var_31}
    var_33 = {var_7: var_32}
    var_34 = []
    var_35 = '--min-confidence'
    var_36 = '50'
    var_37 = [var_3, var_35, var_36]
    var_38 = b'invalid toml content'
    var_39 = []
    var_40 = []
    var_41 = None
    var_42 = module_0.make_config(var_40, var_41)
    var_43 = b'[tool.vulture]\nunknown_key = true'
    var_44 = []
    var_45 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_46 = []
    var_47 = '--min-confidence'
    var_48 = 'not_an_int'
    var_49 = [var_47, var_48]
    var_50 = None
    var_51 = module_0.make_config(var_49, var_50)
    var_52 = '[tool.vulture]\nmin_confidence = 20\n'
    var_53 = '--config'
    var_54 = None
    var_55 = module_0.make_config(var_48, var_54)
    var_56 = [var_50]



# Parsed testcases at query #37
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    paths = ["src"]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    '
    var_9 = 'custom_path'
    var_10 = '90'
    var_11 = [var_9, var_4, var_10]
    var_12 = 'path'
    var_13 = '--make-whitelist'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--verbose'
    var_17 = [var_16]
    var_18 = '--sort-by-size'
    var_19 = [var_12, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--ignore-decorators'
    var_22 = 'dec1,dec2'
    var_23 = '--ignore-names'
    var_24 = 'name1,name2'
    var_25 = [var_12, var_21, var_22, var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--exclude'
    var_28 = 'file1.py,file2.py'
    var_29 = [var_12, var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--config'
    var_32 = 'custom.toml'
    var_33 = [var_12, var_31, var_32]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)
    var_37 = '--verbose'
    var_38 = [var_37]
    var_39 = module_0.make_config(var_38)
    var_40 = b''
    var_41 = module_0.make_config(tomlfile=var_6)



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '--verbose'
    var_9 = '--sort-by-size'
    var_10 = 'test.py'
    var_11 = [var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 10\n    paths = ["path1", "path2"]\n    '
    var_14 = []
    var_15 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1", "path2"]\n    '
    var_16 = '90'
    var_17 = 'cli_path'
    var_18 = [var_2, var_16, var_17]
    var_19 = b'\n    [tool.vulture]\n    exclude = ["file*.py"]\n    min_confidence = 10\n    '
    var_20 = [var_9, var_10]
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = [var_10]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #39
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    paths = ["src"]\n    exclude = ["test*.py"]\n    min_confidence = 20\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    paths = ["src"]\n    min_confidence = 20\n    '
    var_13 = '80'
    var_14 = [var_5, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_21 = []
    var_22 = 'pyproject.toml'
    var_23 = '\n            [tool.vulture]\n            paths = ["src"]\n            exclude = ["test*.py"]\n            '
    var_24 = '--config'
    var_25 = module_0.make_config(var_17)
    var_26 = '\n    [tool.vulture]\n    paths = ["src"]\n    verbose = true\n    '
    var_27 = []



# Parsed testcases at query #40
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    '
    var_12 = [var_4, var_5, var_6]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    paths = []\n    '
    var_20 = []
    var_21 = '\n    [tool.vulture]\n    paths = ["src"]\n    '
    var_22 = []



# Parsed testcases at query #41
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '80'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = 'custom.py'
    var_12 = '90'
    var_13 = [var_11, var_4, var_12]
    var_14 = []
    var_15 = ''
    var_16 = module_0.make_config(var_14, var_3)
    var_17 = []
    var_18 = '[tool.vulture]\ninvalid_key = true'
    var_19 = module_0.make_config(var_17, var_3)
    var_20 = []
    var_21 = "[tool.vulture]\nverbose = 'yes'"
    var_22 = module_0.make_config(var_20, var_3)
    var_23 = '[tool.vulture]\nmin_confidence = 50\n'
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = '--config'
    var_27 = module_0.make_config(var_3)
    var_28 = []
    var_29 = '[tool.vulture]\npaths = []'
    var_30 = module_0.make_config(var_28, var_3)



# Parsed testcases at query #42
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\npaths = ["toml_path1", "toml_path2"]\nmin_confidence = 30\nexclude = ["exclude1", "exclude2"]\n'
    var_11 = []
    var_12 = '\n[tool.vulture]\npaths = ["toml_path"]\nmin_confidence = 30\n'
    var_13 = 'cli_path'
    var_14 = '80'
    var_15 = [var_13, var_5, var_14]
    var_16 = '--exclude'
    var_17 = 'file1.py,file2.py'
    var_18 = '--ignore-decorators'
    var_19 = 'deco1,deco2'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = []
    var_23 = None
    var_24 = module_0.make_config(var_22, var_23)
    var_25 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_26 = 'path'
    var_27 = [var_26]
    var_28 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_29 = 'path'
    var_30 = [var_29]
    var_31 = '\n[tool.vulture]\npaths = ["path"]\nverbose = true\n'
    var_32 = []



# Parsed testcases at query #43
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_6 = 'utf-8'
    var_7 = [var_2]
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = [var_2, var_8, var_9]
    var_11 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_12 = 'test.py'
    var_13 = [var_12]
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_15 = 'test.py'
    var_16 = [var_15]
    var_17 = '\n    [tool.vulture]\n    verbose = true\n    '
    var_18 = []
    var_19 = '--version'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--help'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #44
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n    [tool.vulture]\n    paths = ["src", "tests"]\n    min_confidence = 80\n    exclude = ["*.pyc"]\n    ignore_decorators = ["@app.route"]\n    ignore_names = ["_private"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    '
    var_11 = []
    var_12 = b'\n    [tool.vulture]\n    paths = ["src"]\n    min_confidence = 80\n    '
    var_13 = 'cli_path'
    var_14 = '20'
    var_15 = [var_13, var_5, var_14]
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["src"]\n    '
    var_20 = []
    var_21 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["src"]\n    '
    var_22 = []
    var_23 = b'\n    [tool.vulture]\n    min_confidence = 80\n    '
    var_24 = 'some_path'
    var_25 = [var_24]
    var_26 = 'pyproject.toml'
    var_27 = '[tool.vulture]\npaths = ["test_path"]\nmin_confidence = 30\n'
    var_28 = []
    var_29 = module_0.make_config(var_28)
    var_30 = 'custom.toml'
    var_31 = '[tool.vulture]\npaths = ["custom_path"]\n'
    var_32 = '--config'
    var_33 = module_0.make_config(var_18)



# Parsed testcases at query #45
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = 'custom_path'
    var_12 = '80'
    var_13 = [var_11, var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = 'path'
    var_16 = [var_15]
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = []
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #46
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_10 = []
    var_11 = '70'
    var_12 = [var_2, var_11]
    var_13 = 'cli_path.py'
    var_14 = [var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = true\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "high"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #47
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_1 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    min_confidence = 5\n    '
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'cli_path'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'test_path'
    var_7 = [var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_12 = 'test_path'
    var_13 = [var_12]
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_15 = 'test_path'
    var_16 = [var_15]
    var_17 = '--verbose'
    var_18 = [var_17, var_6]
    var_19 = module_0.make_config(var_18)
    var_20 = '\n        [tool.vulture]\n        paths = ["from_config_file"]\n        '
    var_21 = '--config'
    var_22 = module_0.make_config(var_16)



# Parsed testcases at query #48
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--paths'
    var_5 = 'test1.py'
    var_6 = 'test2.py'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = ''
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_14 = '\n    [tool.vulture]\n    min_confidence = "invalid"\n    paths = ["test.py"]\n    '
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '--version'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #49
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 80\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["test_*.py"]\n    sort_by_size = true\n    '
    var_10 = []
    var_11 = 'cli_path'
    var_12 = '90'
    var_13 = [var_11, var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = 'path'
    var_16 = [var_15]
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = 'path'
    var_23 = '--config'
    var_24 = 'nonexistent.toml'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--exclude'
    var_28 = 'test_*.py,*.bak'
    var_29 = '--ignore-decorators'
    var_30 = '@app.route,@decorator'
    var_31 = '--ignore-names'
    var_32 = 'private_*,_internal'
    var_33 = '--make-whitelist'
    var_34 = '--sort-by-size'
    var_35 = '75'
    var_36 = [var_22, var_27, var_28, var_29, var_30, var_31, var_32, var_33, var_34, var_4, var_35]
    var_37 = module_0.make_config(var_36)
    var_38 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["test_path"]\n    '
    var_39 = []
    var_40 = '\n                [tool.vulture]\n                min_confidence = 30\n                paths = ["from_default_config"]\n                '
    var_41 = []
    var_42 = module_0.make_config(var_41)
    var_43 = '\n                [tool.vulture]\n                min_confidence = 30\n                paths = ["from_default_config"]\n                '
    var_44 = 'cli_path'
    var_45 = '--min-confidence'
    var_46 = '45'
    var_47 = [var_44, var_45, var_46]
    var_48 = module_0.make_config(var_47)



# Parsed testcases at query #50
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = '--exclude'
    var_9 = 'test_*.py,venv'
    var_10 = '--ignore-decorators'
    var_11 = '@app.route,@require_*'
    var_12 = '--ignore-names'
    var_13 = 'visit_*,do_*'
    var_14 = '--make-whitelist'
    var_15 = '--sort-by-size'
    var_16 = '--verbose'
    var_17 = [var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_2]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\npaths = ["toml_path1.py", "toml_path2.py"]\nmin_confidence = 10\nexclude = ["excluded*.py"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nsort_by_size = true\nverbose = true\n'
    var_20 = []
    var_21 = '20'
    var_22 = 'cli_path.py'
    var_23 = [var_6, var_21, var_22]
    var_24 = '\n[tool.vulture]\ninvalid_key = "value"\npaths = ["path.py"]\n'
    var_25 = []
    var_26 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\npaths = ["path.py"]\n'
    var_27 = []
    var_28 = '--min-confidence'
    var_29 = 'not_an_int'
    var_30 = 'path.py'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--unknown-option'
    var_34 = 'path.py'
    var_35 = [var_33, var_34]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #51
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '75'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    paths = ["src/"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 50\n    paths = ["src/"]\n    '
    var_12 = '90'
    var_13 = [var_4, var_12]
    var_14 = '--make-whitelist'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '--sort-by-size'
    var_18 = [var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--ignore-decorators'
    var_21 = '@app.route,@require_*'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--ignore-names'
    var_25 = 'visit_*,do_*'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--exclude'
    var_29 = '*settings.py,docs,*/test_*.py,venv'
    var_30 = [var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = []
    var_33 = module_0.make_config(var_32)
    var_34 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_35 = 'test.py'
    var_36 = [var_35]
    var_37 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_38 = 'test.py'
    var_39 = [var_38]



# Parsed testcases at query #52
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    verbose = false\n    '
    var_10 = '75'
    var_11 = [var_2, var_10]
    var_12 = []
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'\n    [tool.vulture]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    paths = ["path"]\n    '
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["@app.route", "@require_*"]\n    ignore_names = ["visit_*", "do_*"]\n    paths = ["path"]\n    '
    var_18 = []
    var_19 = '--exclude'
    var_20 = 'file1.py,file2.py'
    var_21 = 'path'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--ignore-decorators'
    var_25 = '@dec1,@dec2'
    var_26 = '--ignore-names'
    var_27 = 'name1,name2'
    var_28 = [var_24, var_25, var_26, var_27, var_21]
    var_29 = module_0.make_config(var_28)
    var_30 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path"]\n    '
    var_31 = []
    var_32 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_33 = []



# Parsed testcases at query #53
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 75\n    exclude = ["test_*.py", "docs"]\n    verbose = true\n    '
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    paths = ["toml_path"]\n    min_confidence = 10\n    '
    var_12 = 'cli_path'
    var_13 = '90'
    var_14 = [var_12, var_4, var_13]
    var_15 = b'\n    [tool.vulture]\n    invalid_key = true\n    '
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = b'\n        [tool.vulture]\n        paths = ["custom_path"]\n        '
    var_21 = '--config'
    var_22 = module_0.make_config(var_18)
    var_23 = 'path'
    var_24 = '--exclude'
    var_25 = 'a.py,b.py'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--ignore-decorators'
    var_29 = '@deco1,@deco2'
    var_30 = '--ignore-names'
    var_31 = 'name1,name2'
    var_32 = [var_23, var_28, var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--make-whitelist'
    var_35 = '--sort-by-size'
    var_36 = [var_23, var_34, var_35]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #54
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\npaths = ["src", "tests"]\n'
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\n'
    var_9 = '80'
    var_10 = [var_1, var_9]
    var_11 = '\n[tool.vulture]\ninvalid_key = 10\n'
    var_12 = '\n[tool.vulture]\nmin_confidence = "high"\n'
    var_13 = '--exclude'
    var_14 = 'foo'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '--exclude'
    var_18 = 'a,b,c'
    var_19 = 'path'
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '\n[tool.vulture]\nverbose = true\npaths = ["src"]\n'



# Parsed testcases at query #55
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = 'test_file.py'
    var_8 = 'another_file.py'
    var_9 = [var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = b'\n[tool.vulture]\npaths = ["path1", "path2"]\nmin_confidence = 30\nverbose = true\n'
    var_12 = []
    var_13 = b'\n[tool.vulture]\nmin_confidence = 30\n'
    var_14 = '--min-confidence'
    var_15 = '70'
    var_16 = [var_14, var_15]
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = b'\n[tool.vulture]\nunknown_key = "value"\n'
    var_20 = []
    var_21 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_22 = []
    var_23 = 'custom.toml'
    var_24 = '[tool.vulture]\nmin_confidence = 25\n'
    var_25 = '--config'
    var_26 = 'test.py'
    var_27 = module_0.make_config(var_4)
    var_28 = 'test.py'
    var_29 = '--exclude'
    var_30 = '*.pyc,test_*.py'
    var_31 = '--ignore-decorators'
    var_32 = '@app.route,@login_required'
    var_33 = '--ignore-names'
    var_34 = 'private_*,_internal'
    var_35 = '--make-whitelist'
    var_36 = '75'
    var_37 = '--sort-by-size'
    var_38 = [var_28, var_29, var_30, var_31, var_32, var_33, var_34, var_35, var_25, var_36, var_37, var_4]
    var_39 = module_0.make_config(var_38)
    var_40 = b'\n[tool.vulture]\npaths = ["test_path"]\n'
    var_41 = '--verbose'
    var_42 = [var_41]



# Parsed testcases at query #56
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = '--verbose'
    var_7 = '--sort-by-size'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_12 = '80'
    var_13 = 'cli_path'
    var_14 = [var_2, var_12, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path"]\n    '
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_19 = '--version'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--help'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #57
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
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_6, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = b'\n    [tool.vulture]\n    unknown_key = true\n    paths = ["path1"]\n    '
    var_19 = []



# Parsed testcases at query #58
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_3 = []
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--paths'
    var_7 = 'cli_path'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = '30'
    var_10 = 'path1'
    var_11 = 'path2'
    var_12 = [var_4, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #59
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'file1.py'
    var_6 = 'dir/'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n[tool.vulture]\npaths = ["src/"]\nmin_confidence = 75\nexclude = ["test_*", "docs"]\nignore_decorators = ["@app.route"]\nignore_names = ["internal_*"]\nmake_whitelist = true\nsort_by_size = true\nverbose = true\n'
    var_10 = []
    var_11 = b'[tool.vulture]\npaths = ["src/"]\nmin_confidence = 75\n'
    var_12 = '25'
    var_13 = 'myfile.py'
    var_14 = [var_2, var_12, var_13]
    var_15 = b'[tool.vulture]\ninvalid_key = true\n'
    var_16 = []
    var_17 = b'[tool.vulture]\nmin_confidence = "high"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--config'
    var_22 = 'nonexistent.toml'
    var_23 = 'file.py'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = b'[tool.vulture]\nmin_confidence = 90\n'
    var_27 = 'file.py'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28)



# Parsed testcases at query #60
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_3, var_4]
    var_6 = None
    var_7 = module_0.make_config(var_5, var_6)
    var_8 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 50\n    exclude = ["exclude1", "exclude2"]\n    ignore_decorators = ["dec1", "dec2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    '
    var_9 = []
    var_10 = 'cli_path'
    var_11 = '--min-confidence'
    var_12 = '75'
    var_13 = [var_10, var_11, var_12]
    var_14 = '\n    [tool.vulture]\n    paths = ["test"]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    paths = ["test"]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = '--version'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19, var_6)
    var_21 = '--help'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22, var_6)
    var_24 = 'test.py'
    var_25 = '--config'
    var_26 = 'nonexistent.toml'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27, var_6)



# Parsed testcases at query #61
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py", "docs"]\n    ignore_decorators = ["@app.route"]\n    ignore_names = ["private_*"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    paths = ["src", "lib"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = '--paths'
    var_13 = 'custom_path'
    var_14 = [var_2, var_11, var_12, var_13]
    var_15 = []
    var_16 = '--min-confidence'
    var_17 = '50'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test"]\n    '
    var_21 = []
    var_22 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test"]\n    '
    var_23 = []
    var_24 = '--config'
    var_25 = module_0.make_config(var_17)
    var_26 = [var_4]



# Parsed testcases at query #62
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = 'cli_path'
    var_6 = [var_5]
    var_7 = '--min-confidence'
    var_8 = '20'
    var_9 = '--verbose'
    var_10 = [var_5, var_7, var_8, var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_16 = []
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #63
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1", "path2"]\n    '
    var_12 = '80'
    var_13 = [var_2, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "string"\n    '
    var_17 = []
    var_18 = '--min-confidence'
    var_19 = '50'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #64
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--exclude'
    var_3 = 'test.py,foo.py'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 30\n    paths = ["src", "tests"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_4, var_10]
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["test.py"]\n    '
    var_17 = []
    var_18 = 'path1.py'
    var_19 = 'path2.py'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = 'test.py'
    var_23 = '--verbose'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--make-whitelist'
    var_27 = [var_22, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--sort-by-size'
    var_30 = [var_22, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '--ignore-decorators'
    var_33 = '@app.route,@require_*'
    var_34 = '--ignore-names'
    var_35 = 'visit_*,do_*'
    var_36 = [var_32, var_33, var_34, var_35, var_22]
    var_37 = module_0.make_config(var_36)
    var_38 = '--config'
    var_39 = 'nonexistent.toml'
    var_40 = [var_38, var_39, var_22]
    var_41 = module_0.make_config(var_40)



# Parsed testcases at query #65
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 20\n    exclude = ["file*.py", "dir/"]\n    paths = ["src/"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 20\n    '
    var_13 = '80'
    var_14 = [var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)
    var_22 = 'test_file.py'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23, var_20)



# Parsed testcases at query #66
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["file1.py", "file2.py"]\n    paths = ["src"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11]
    var_13 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = 'some_file.py'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #67
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_4 = []
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'script.py'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--verbose'
    var_10 = '--sort-by-size'
    var_11 = 'test.py'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12, var_1)
    var_14 = 'temp_config.toml'
    var_15 = '--config'
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = "[tool.vulture]\nunknown_key = 5\npaths = ['test.py']"
    var_20 = []



# Parsed testcases at query #68
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_4 = 'utf-8'
    var_5 = []
    var_6 = '--exclude'
    var_7 = 'test1.py,test2.py'
    var_8 = '--min-confidence'
    var_9 = '50'
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = [var_6, var_7, var_8, var_9]
    var_12 = module_0.make_config(var_11, var_1)
    var_13 = 'path1'
    var_14 = 'path2'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15, var_1)
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_21 = []
    var_22 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_23 = []
    var_24 = '--min-confidence'
    var_25 = 'not_an_int'
    var_26 = [var_24, var_25]
    var_27 = None
    var_28 = module_0.make_config(var_26, var_27)



# Parsed testcases at query #69
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '80'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '50'
    var_12 = [var_3, var_11]
    var_13 = '--verbose'
    var_14 = [var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path1"]\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_19 = []
    var_20 = '--version'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--help'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)



# Parsed testcases at query #70
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '--exclude'
    var_9 = '*.py,test_*.py'
    var_10 = '--ignore-decorators'
    var_11 = '@app.route,@require_*'
    var_12 = '--ignore-names'
    var_13 = 'visit_*,do_*'
    var_14 = '--make-whitelist'
    var_15 = '--min-confidence'
    var_16 = '50'
    var_17 = '--sort-by-size'
    var_18 = '--verbose'
    var_19 = [var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_4, var_5]
    var_20 = module_0.make_config(var_19)
    var_21 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_22 = []
    var_23 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    paths = ["toml_path"]\n    min_confidence = 10\n    '
    var_24 = '*.py'
    var_25 = '20'
    var_26 = 'cli_path'
    var_27 = [var_8, var_24, var_15, var_25, var_26]
    var_28 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_29 = []
    var_30 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_31 = []
    var_32 = '--min-confidence'
    var_33 = 'not_an_int'
    var_34 = 'path1'
    var_35 = [var_32, var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = '--unknown-arg'
    var_38 = 'path1'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '--config'
    var_42 = 'non_existent.toml'
    var_43 = [var_41, var_42, var_38]
    var_44 = module_0.make_config(var_43)
    var_45 = []
    var_46 = module_0.make_config(var_45)



# Parsed testcases at query #71
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 20\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = '--exclude'
    var_14 = 'custom*.py'
    var_15 = '--no-make-whitelist'
    var_16 = [var_13, var_14, var_15]
    var_17 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_18 = []
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_20 = []
    var_21 = []
    var_22 = ''
    var_23 = module_0.make_config(var_21, var_3)
    var_24 = '/some/path'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = "[tool.vulture]\npaths = ['/toml/path']"
    var_28 = []
    var_29 = '--version'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '--help'
    var_33 = [var_32]
    var_34 = module_0.make_config(var_33)



# Parsed testcases at query #72
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
    var_7 = '@app.route,@require_*'
    var_8 = '--ignore-names'
    var_9 = 'visit_*,do_*'
    var_10 = '--make-whitelist'
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = '--sort-by-size'
    var_14 = '--verbose'
    var_15 = [var_0, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_18 = []
    var_19 = 'cli_path'
    var_20 = '90'
    var_21 = [var_19, var_11, var_20]
    var_22 = '--min-confidence'
    var_23 = '10'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path"]\n    '
    var_27 = []
    var_28 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_29 = []
    var_30 = 'pyproject.toml'
    var_31 = '\n            [tool.vulture]\n            min_confidence = 25\n            paths = ["toml_path"]\n            '
    var_32 = '--config'
    var_33 = module_0.make_config(var_24)
    var_34 = []
    var_35 = module_0.make_config(var_34)
    var_36 = '--verbose'
    var_37 = [var_36]



# Parsed testcases at query #73
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '80'
    var_5 = '--verbose'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    paths = ["src/"]\n    '
    var_9 = '90'
    var_10 = [var_3, var_9]
    var_11 = '--exclude'
    var_12 = 'a.py,b.py,c.py'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '--ignore-decorators'
    var_16 = '@app.route,@require_*'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--ignore-names'
    var_20 = 'visit_*,do_*'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--make-whitelist'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--sort-by-size'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--config'
    var_30 = 'custom.toml'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #74
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_4, var_11, var_6]
    var_13 = '--min-confidence'
    var_14 = '50'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["src"]\n    '
    var_18 = []
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["src"]\n    '
    var_20 = []



# Parsed testcases at query #75
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test.py'
    var_6 = 'dir/'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '30'
    var_12 = [var_2, var_11, var_4]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = b'[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []



# Parsed testcases at query #76
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = module_0.make_config(var_3)
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--exclude'
    var_8 = 'test1.py,test2.py'
    var_9 = '--verbose'
    var_10 = '--make-whitelist'
    var_11 = '--sort-by-size'
    var_12 = '--ignore-decorators'
    var_13 = '@deco1,@deco2'
    var_14 = '--ignore-names'
    var_15 = 'name1,name2'
    var_16 = [var_2, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n    [tool.vulture]\n    exclude = ["config.py", "settings.py"]\n    min_confidence = 30\n    verbose = true\n    paths = ["path1.py", "path2.py"]\n    '
    var_19 = 'utf-8'
    var_20 = []
    var_21 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["path1.py"]\n    verbose = false\n    '
    var_22 = 'cli_path.py'
    var_23 = '70'
    var_24 = [var_22, var_5, var_23, var_9]
    var_25 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1.py"]\n    '
    var_26 = []
    var_27 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1.py"]\n    '
    var_28 = []
    var_29 = 'test.py'
    var_30 = '--min-confidence'
    var_31 = 'not_an_int'
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = 'test.py'
    var_35 = '--unknown-flag'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36)
    var_38 = '\n    [tool.vulture]\n    min_confidence = 30\n    '
    var_39 = []
    var_40 = '\n    [tool.vulture]\n    paths = []\n    '
    var_41 = []



# Parsed testcases at query #77
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    min_confidence = 50\n    '
    var_3 = 'utf-8'
    var_4 = '\n    [tool.vulture]\n    min_confidence = 50\n    sort_by_size = true\n    '
    var_5 = '--min-confidence'
    var_6 = '80'
    var_7 = 'path/to/file.py'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--exclude'
    var_10 = 'test*.py,*.bak'
    var_11 = 'src'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--config'
    var_15 = 'nonexistent.toml'
    var_16 = 'file.py'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_20 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_21 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    '
    var_22 = '\n    [tool.vulture]\n    verbose = true\n    '
    var_23 = [var_16]



# Parsed testcases at query #78
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1.py'
    var_4 = 'path2.py'
    var_5 = '--min-confidence'
    var_6 = '10'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'cli_path.py'
    var_13 = '20'
    var_14 = [var_12, var_5, var_13]
    var_15 = []
    var_16 = b''
    var_17 = module_0.make_config(var_15, var_3)
    var_18 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19, var_3)
    var_21 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_22 = []
    var_23 = module_0.make_config(var_22, var_3)



# Parsed testcases at query #79
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'test.py'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n    [tool.vulture]\n    paths = ["test1.py", "test2.py"]\n    min_confidence = 30\n    exclude = ["test_exclude.py"]\n    '
    var_8 = []
    var_9 = 'cli.py'
    var_10 = '70'
    var_11 = [var_9, var_3, var_10]
    var_12 = [var_2]
    var_13 = module_0.make_config(var_12)
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_19 = []
    var_20 = 'test.py'
    var_21 = '--min-confidence'
    var_22 = 'not_an_int'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--verbose'
    var_26 = [var_25]
    var_27 = '\n        [tool.vulture]\n        paths = ["test.py"]\n        min_confidence = 25\n        '
    var_28 = '--config'
    var_29 = '\n        [tool.vulture]\n        paths = ["test.py"]\n        '
    var_30 = '--exclude'
    var_31 = 'file1.py,file2.py'
    var_32 = '--ignore-decorators'
    var_33 = 'deco1,deco2'
    var_34 = '--ignore-names'
    var_35 = 'name1,name2'
    var_36 = '--make-whitelist'
    var_37 = '40'
    var_38 = '--sort-by-size'
    var_39 = [var_21, var_30, var_31, var_32, var_33, var_34, var_35, var_36, var_22, var_37, var_38, var_25]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #80
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--verbose'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test_*.py", "temp/"]\n    min_confidence = 30\n    sort_by_size = true\n    paths = ["src/", "lib/"]\n    '
    var_9 = '80'
    var_10 = 'custom.py'
    var_11 = [var_2, var_9, var_10]
    var_12 = '\n    [tool.vulture]\n    unknown_key = true\n    '
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_14 = '--verbose'
    var_15 = [var_14]
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #81
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = 'test.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 75\n    paths = ["src", "tests"]\n    exclude = ["*.pyc"]\n    '
    var_10 = []
    var_11 = '90'
    var_12 = [var_3, var_11]
    var_13 = 'custom.py'
    var_14 = [var_13]
    var_15 = '--exclude'
    var_16 = 'file1.py,file2.py'
    var_17 = '--ignore-decorators'
    var_18 = '@deco1,@deco2'
    var_19 = '--ignore-names'
    var_20 = 'name1,name2'
    var_21 = '--make-whitelist'
    var_22 = '--sort-by-size'
    var_23 = '30'
    var_24 = 'path1'
    var_25 = 'path2'
    var_26 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_3, var_23, var_5, var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = []
    var_29 = None
    var_30 = module_0.make_config(var_28, var_29)
    var_31 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_32 = []



# Parsed testcases at query #82
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.make_config(var_2, var_3)
    var_5 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_6 = []
    var_7 = 'cli_path'
    var_8 = '--min-confidence'
    var_9 = '20'
    var_10 = '--verbose'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = []
    var_13 = ''
    var_14 = []
    var_15 = ''
    var_16 = module_0.make_config(var_14, var_2)
    var_17 = '\n    [tool.vulture]\n    unknown_key = true\n    paths = ["path1"]\n    '
    var_18 = []
    var_19 = module_0.make_config(var_18, var_15)
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_21 = []
    var_22 = module_0.make_config(var_21, var_15)



# Parsed testcases at query #83
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = '--verbose'
    var_9 = '--make-whitelist'
    var_10 = 'path'
    var_11 = [var_6, var_7, var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_14 = []
    var_15 = '90'
    var_16 = 'cli_path'
    var_17 = [var_6, var_15, var_16]
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test"]\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test"]\n    '
    var_21 = []
    var_22 = '\n        [tool.vulture]\n        paths = ["config_file_path"]\n        '
    var_23 = '--config'
    var_24 = module_0.make_config(var_3)



# Parsed testcases at query #84
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_3 = []
    var_4 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["from_toml"]\n    '
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'from_cli'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--verbose'
    var_10 = 'test.py'
    var_11 = [var_5, var_6, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '--config'
    var_14 = 'nonexistent.toml'
    var_15 = [var_13, var_14, var_10]
    var_16 = module_0.make_config(var_15)
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_20 = 'test.py'
    var_21 = [var_20]



# Parsed testcases at query #85
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '--min-confidence'
    var_7 = '50'
    var_8 = '--exclude'
    var_9 = 'test_*.py,docs'
    var_10 = '--ignore-decorators'
    var_11 = '@app.route,@require_*'
    var_12 = '--ignore-names'
    var_13 = 'visit_*,do_*'
    var_14 = '--make-whitelist'
    var_15 = '--sort-by-size'
    var_16 = '--verbose'
    var_17 = [var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_2]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1.py", "path2.py"]\n'
    var_20 = []
    var_21 = '80'
    var_22 = 'custom_path.py'
    var_23 = [var_6, var_21, var_22]
    var_24 = '\n[tool.vulture]\ninvalid_key = true\npaths = ["path1.py"]\n'
    var_25 = []
    var_26 = '\n[tool.vulture]\nmin_confidence = "high"\npaths = ["path1.py"]\n'
    var_27 = []
    var_28 = '\n[tool.vulture]\nmin_confidence = 10\n'
    var_29 = []
    var_30 = '--config'
    var_31 = module_0.make_config(var_3)
    var_32 = [var_16]
    var_33 = '--config'
    var_34 = 'non_existent.toml'
    var_35 = 'test_path.py'
    var_36 = [var_33, var_34, var_35]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #86
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    verbose = true\n    paths = ["src/"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 75\n    paths = ["src/"]\n    '
    var_12 = '90'
    var_13 = [var_2, var_12]
    var_14 = '--exclude'
    var_15 = 'test_*.py,docs'
    var_16 = '--ignore-decorators'
    var_17 = '@app.route,@require_*'
    var_18 = '--ignore-names'
    var_19 = 'visit_*,do_*'
    var_20 = '--make-whitelist'
    var_21 = '--sort-by-size'
    var_22 = '--config'
    var_23 = 'custom.toml'
    var_24 = 'path1'
    var_25 = 'path2'
    var_26 = [var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--min-confidence'
    var_29 = '10'
    var_30 = [var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["file.py"]\n    '
    var_33 = []
    var_34 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["file.py"]\n    '
    var_35 = []



# Parsed testcases at query #87
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--config'
    var_3 = 'nonexistent.toml'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '--verbose'
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = 'test_path'
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_13 = []
    var_14 = '80'
    var_15 = [var_7, var_14, var_6]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = '--verbose'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #88
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = '80'
    var_10 = [var_3, var_9]
    var_11 = '--min-confidence'
    var_12 = '50'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path1"]\n    '
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '



# Parsed testcases at query #89
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = 'cli_path'
    var_12 = [var_2, var_10, var_11]
    var_13 = '\n    [tool.vulture]\n    exclude = ["file*.py"]\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '\n    [tool.vulture]\n    paths = ["path1"]\n    '
    var_22 = []



# Parsed testcases at query #90
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 10\n    sort_by_size = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_10 = []
    var_11 = 'io'
    var_12 = __import__(var_11)
    var_13 = b'\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_14 = '80'
    var_15 = [var_2, var_14]
    var_16 = __import__(var_11)
    var_17 = 'pyproject.toml'
    var_18 = '\n    [tool.vulture]\n    min_confidence = 30\n    '
    var_19 = '--config'
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = b'\n    [tool.vulture]\n    unknown_key = true\n    '
    var_23 = []
    var_24 = 'io'
    var_25 = __import__(var_24)
    var_26 = module_0.make_config(var_23, var_4)



# Parsed testcases at query #91
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n[tool.vulture]\nmin_confidence = 75\nexclude = ["test*.py", "docs/"]\nignore_decorators = ["@app.route"]\nignore_names = ["private_*"]\nmake_whitelist = true\nsort_by_size = true\nverbose = true\npaths = ["src", "tests"]\n'
    var_11 = []
    var_12 = '20'
    var_13 = 'path3'
    var_14 = [var_5, var_12, var_7, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n[tool.vulture]\nmin_confidence = "invalid"\n'
    var_19 = []
    var_20 = '\n[tool.vulture]\nunknown_key = true\n'
    var_21 = []



# Parsed testcases at query #92
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--verbose'
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.make_config(var_4, var_5)
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = 'path1'
    var_10 = 'path2'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_14 = []
    var_15 = '80'
    var_16 = '--verbose'
    var_17 = [var_7, var_15, var_16]
    var_18 = '\n    [tool.vulture]\n    invalid_key = 5\n    '
    var_19 = []
    var_20 = []
    var_21 = '[tool.vulture]\n'
    var_22 = module_0.make_config(var_20, var_7)
    var_23 = "[tool.vulture]\npaths = ['test.py']\n"
    var_24 = '--config'
    var_25 = 'non_existent_file.toml'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--version'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--help'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #93
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_2 = b'[tool.vulture]\nmin_confidence = 10\npaths = ["toml_path"]'
    var_3 = '--min-confidence'
    var_4 = '20'
    var_5 = '--verbose'
    var_6 = 'cli_path.py'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = '--exclude'
    var_9 = 'test_*.py,docs'
    var_10 = '--sort-by-size'
    var_11 = 'file.py'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = 'custom_config.toml'
    var_15 = '[tool.vulture]\nmin_confidence = 50\n'
    var_16 = '--config'
    var_17 = 'some_path.py'
    var_18 = module_0.make_config(var_12)
    var_19 = '--verbose'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = b'[tool.vulture]\nunknown_key = true\n'
    var_23 = 'path.py'
    var_24 = [var_23]
    var_25 = b'[tool.vulture]\nmin_confidence = "10"\n'
    var_26 = 'path.py'
    var_27 = [var_26]



# Parsed testcases at query #94
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 50\n    paths = ["src"]\n    '
    var_1 = '--verbose'
    var_2 = [var_1]
    var_3 = '--min-confidence'
    var_4 = '80'
    var_5 = [var_3, var_4, var_1]
    var_6 = 'test_path'
    var_7 = [var_6]
    var_8 = ''
    var_9 = '\n        [tool.vulture]\n        paths = ["test_dir"]\n        ignore_names = ["_private"]\n        '
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = '--config'
    var_13 = module_0.make_config(var_3)
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'file1.py,file2.py'
    var_18 = 'src'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '\n    [tool.vulture]\n    unknown_key = true\n    '
    var_22 = 'test_path'
    var_23 = [var_22]



# Parsed testcases at query #95
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n    [tool.vulture]\n    exclude = ["test_pattern"]\n    min_confidence = 50\n    paths = ["test_path"]\n    '
    var_1 = 'config.toml'
    var_2 = '--verbose'
    var_3 = [var_2]
    var_4 = '--min-confidence'
    var_5 = '75'
    var_6 = '--verbose'
    var_7 = [var_4, var_5, var_6]
    var_8 = []
    var_9 = 'test_path'
    var_10 = [var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n    [tool.vulture]\n    nonexistent_key = true\n    paths = ["test_path"]\n    '
    var_15 = 'bad_config.toml'
    var_16 = []



# Parsed testcases at query #96
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1.py"]\n    verbose = false\n    '
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'path1.py'
    var_6 = 'path2.py'
    var_7 = [var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = []
    var_10 = module_0.make_config(var_9)
    var_11 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_12 = '--verbose'
    var_13 = [var_12]
    var_14 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_15 = '--verbose'
    var_16 = [var_15]
    var_17 = b"[tool.vulture]\nverbose = true\npaths = ['test.py']"
    var_18 = []



# Parsed testcases at query #97
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_3 = []
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_4, var_5, var_6]
    var_8 = '--paths'
    var_9 = 'test.py'
    var_10 = '80'
    var_11 = [var_8, var_9, var_4, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = b'unknown_key = 1'
    var_14 = []
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_18 = []



# Parsed testcases at query #98
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'override_path'
    var_13 = '30'
    var_14 = [var_12, var_5, var_13]
    var_15 = b'[tool.vulture]\ninvalid_key = 5\n'
    var_16 = []
    var_17 = b"[tool.vulture]\nmin_confidence = 'high'\n"
    var_18 = []
    var_19 = []
    var_20 = b'[tool.vulture]\n'
    var_21 = module_0.make_config(var_19, var_3)



# Parsed testcases at query #99
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '80'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 50\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 50\n    paths = ["src"]\n    '
    var_12 = 'custom_path'
    var_13 = '90'
    var_14 = [var_12, var_4, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = '--min-confidence'
    var_20 = '50'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #100
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    paths = ["src/"]\n    '
    var_9 = []
    var_10 = '30'
    var_11 = [var_2, var_10]
    var_12 = '--help'
    var_13 = [var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '--version'
    var_16 = [var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_21 = []
    var_22 = []
    var_23 = module_0.make_config(var_22)
    var_24 = 'cli_path.py'
    var_25 = [var_24]
    var_26 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["src/"]\n    '
    var_27 = []
    var_28 = '--sort-by-size'
    var_29 = '--make-whitelist'
    var_30 = 'file.py'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--exclude'
    var_34 = 'test_*.py,docs'
    var_35 = '--ignore-decorators'
    var_36 = '@app.route'
    var_37 = '--ignore-names'
    var_38 = 'visit_*'
    var_39 = [var_33, var_34, var_35, var_36, var_37, var_38, var_30]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #101
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_4 = [var_0]
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_0, var_5, var_6]
    var_8 = []
    var_9 = module_0.make_config(var_8)
    var_10 = b'[tool.vulture]\nmin_confidence = 20\n'
    var_11 = []
    var_12 = b'[tool.vulture]\ninvalid_key = 1\n'
    var_13 = 'test.py'
    var_14 = [var_13]
    var_15 = b"[tool.vulture]\nmin_confidence = 'high'\n"
    var_16 = 'test.py'
    var_17 = [var_16]
    var_18 = 'test.py'
    var_19 = '--min-confidence'
    var_20 = 'high'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = 'test.py'
    var_24 = '--unknown-arg'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--verbose'
    var_28 = [var_23, var_27]



# Parsed testcases at query #102
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["file1.py", "file2.py"]\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = '70'
    var_12 = [var_5, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "string"\n    '
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '-v'
    var_21 = [var_20]



# Parsed testcases at query #103
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test_file.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["exclude1.py", "exclude2.py"]\n    min_confidence = 30\n    paths = ["path1.py", "path2.py"]\n    '
    var_9 = 'pyproject.toml'
    var_10 = []
    var_11 = '--min-confidence'
    var_12 = '80'
    var_13 = [var_11, var_12]
    var_14 = '--verbose'
    var_15 = [var_14]
    var_16 = 'custom_config.toml'
    var_17 = '[tool.vulture]\nmin_confidence = 45\n'
    var_18 = '--config'
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_22 = 'bad_config.toml'
    var_23 = '--config'
    var_24 = [var_23, var_20]
    var_25 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_26 = 'wrong_type.toml'
    var_27 = '--config'
    var_28 = [var_27, var_20]
    var_29 = 'test.py'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = 'path1.py'
    var_33 = 'path2.py'
    var_34 = [var_32, var_33, var_4]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #104
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1.py'
    var_4 = 'path2.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_13 = 'cli_path'
    var_14 = '20'
    var_15 = [var_13, var_5, var_14]
    var_16 = []
    var_17 = b'[tool.vulture]\nmin_confidence = 10'
    var_18 = module_0.make_config(var_16, var_3)
    var_19 = []
    var_20 = b'[tool.vulture]\nunknown_key = 10'
    var_21 = module_0.make_config(var_19, var_3)
    var_22 = []
    var_23 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_24 = module_0.make_config(var_22, var_3)
    var_25 = '--version'
    var_26 = [var_25]
    var_27 = module_0.make_config(var_26, var_23)
    var_28 = '--help'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_23)
    var_31 = b'[tool.vulture]\nmin_confidence = 30\npaths = ["test_path"]'
    var_32 = []
    var_33 = None
    var_34 = module_0.make_config(var_32, var_33)
    var_35 = '--config'
    var_36 = module_0.make_config(var_24, var_33)
    var_37 = '--config'
    var_38 = 'nonexistent.toml'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39, var_33)
    var_41 = []
    var_42 = None
    var_43 = module_0.make_config(var_41, var_42)
    var_44 = [var_13]
    var_45 = b'[tool.vulture]\nmin_confidence = 5'
    var_46 = 'path1'
    var_47 = 'path2'
    var_48 = '--exclude'
    var_49 = '*.py,test_*'
    var_50 = '--ignore-names'
    var_51 = 'foo,bar'
    var_52 = '--ignore-decorators'
    var_53 = '@deco1,@deco2'
    var_54 = '--make-whitelist'
    var_55 = '--sort-by-size'
    var_56 = [var_46, var_47, var_48, var_49, var_50, var_51, var_52, var_53, var_54, var_55]
    var_57 = module_0.make_config(var_56, var_42)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    sort_by_size = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_4, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)
    var_22 = '--config'
    var_23 = 'nonexistent.toml'
    var_24 = [var_20, var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--exclude'
    var_27 = 'a.py,b.py'
    var_28 = '--ignore-decorators'
    var_29 = '@app.route,@require_*'
    var_30 = [var_20, var_26, var_27, var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '--make-whitelist'
    var_33 = '--sort-by-size'
    var_34 = [var_20, var_32, var_33]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    paths = ["src", "tests"]\n    min_confidence = 80\n    exclude = ["*.pyc"]\n    ignore_decorators = ["@staticmethod"]\n    ignore_names = ["_private"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '30'
    var_14 = [var_12, var_5, var_13]
    var_15 = []
    var_16 = 'rb'
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '--config'
    var_21 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_22 = 'path'
    var_23 = [var_22]
    var_24 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_25 = 'path'
    var_26 = [var_25]
    var_27 = 'path'
    var_28 = '--exclude'
    var_29 = '*.pyc,*.pyo'
    var_30 = '--ignore-decorators'
    var_31 = '@a,@b'
    var_32 = [var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.make_config(var_32, var_26)
    var_34 = '\n    [tool.vulture]\n    paths = ["src"]\n    verbose = true\n    '
    var_35 = []



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_7 = 'path3'
    var_8 = '30'
    var_9 = [var_7, var_2, var_8]
    var_10 = []
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path1"]\n    '
    var_14 = []
    var_15 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_16 = []
    var_17 = 'pyproject.toml'
    var_18 = '--verbose'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #4
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = '90'
    var_10 = [var_3, var_9]
    var_11 = '[tool.vulture]\ninvalid_key = "value"'
    var_12 = '[tool.vulture]\nmin_confidence = "high"'
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '[tool.vulture]\npaths = ["test.py"]'
    var_16 = '--version'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #5
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_8 = '75'
    var_9 = [var_1, var_8]
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_13 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_14 = '--min-confidence'
    var_15 = 'not_an_int'
    var_16 = 'path1'
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = 'pyproject.toml'
    var_20 = module_0.make_config()



# Parsed testcases at query #6
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    paths = ["src"]\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["src"]\n    '
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["src"]\n    '
    var_15 = []
    var_16 = '--min-confidence'
    var_17 = '10'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = []
    var_21 = '--verbose'
    var_22 = [var_21]
    var_23 = 'cli_path.py'
    var_24 = [var_23]
    var_25 = '--exclude'
    var_26 = 'test_*.py,venv'
    var_27 = [var_25, var_26, var_19]
    var_28 = module_0.make_config(var_27)
    var_29 = '--ignore-decorators'
    var_30 = '@app.route'
    var_31 = '--ignore-names'
    var_32 = 'helper_*'
    var_33 = [var_29, var_30, var_31, var_32, var_19]
    var_34 = module_0.make_config(var_33)
    var_35 = '--make-whitelist'
    var_36 = '--sort-by-size'
    var_37 = [var_35, var_36, var_19]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["test_*.py"]\n    '
    var_11 = []
    var_12 = b'\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1"]\n    '
    var_13 = '80'
    var_14 = 'cli_path'
    var_15 = [var_4, var_13, var_14]
    var_16 = b'\n    [tool.vulture]\n    invalid_key = "test"\n    '
    var_17 = []
    var_18 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = b'\n    [tool.vulture]\n    verbose = true\n    paths = ["test_path"]\n    '
    var_23 = []
    var_24 = '--exclude'
    var_25 = 'a.py,b.py,c.py'
    var_26 = 'path'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = '--sort-by-size'
    var_31 = [var_29, var_30, var_26]
    var_32 = module_0.make_config(var_31)
    var_33 = '10'
    var_34 = [var_21, var_33, var_26]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1.py'
    var_1 = 'path2.py'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = []
    var_6 = '--min-confidence'
    var_7 = '20'
    var_8 = 'custom_path.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = "[tool.vulture]\npaths = ['test.py']"
    var_11 = []
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = "[tool.vulture]\nunknown_key = true\npaths = ['test.py']"
    var_15 = []



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_4, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = 'some_path'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--config'



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 20\nexclude = ["test_*.py", "docs"]\nignore_decorators = ["@app.route"]\nignore_names = ["helper_*"]\nmake_whitelist = true\nsort_by_size = true\nverbose = true\npaths = ["src", "lib"]\n'
    var_11 = []
    var_12 = '80'
    var_13 = [var_3, var_12, var_5]
    var_14 = '--config'
    var_15 = module_0.make_config(var_1)
    var_16 = '--min-confidence'
    var_17 = '50'
    var_18 = [var_16, var_17]
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)
    var_21 = '\n[tool.vulture]\ninvalid_key = "value"\npaths = ["src"]\n'
    var_22 = []
    var_23 = '\n[tool.vulture]\nmin_confidence = "high"\npaths = ["src"]\n'
    var_24 = []
    var_25 = '--exclude'
    var_26 = 'test_*.py,docs'
    var_27 = 'path'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--make-whitelist'
    var_31 = '--sort-by-size'
    var_32 = [var_30, var_31, var_27]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #11
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_9 = []
    var_10 = '80'
    var_11 = 'path3'
    var_12 = [var_2, var_10, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = []
    var_15 = []
    var_16 = ''
    var_17 = module_0.make_config(var_15, var_3)
    var_18 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_19 = []



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["src"]\n    exclude = ["tests"]\n    min_confidence = 30\n    verbose = true\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    '
    var_12 = '80'
    var_13 = [var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
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
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '-v'
    var_6 = 'file1.py'
    var_7 = 'dir/'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '20'
    var_13 = 'custom_path.py'
    var_14 = [var_3, var_12, var_13]
    var_15 = '--config'
    var_16 = None
    var_17 = module_0.make_config(var_1, var_16)
    var_18 = '--config'
    var_19 = 'nonexistent.toml'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20, var_1)
    var_22 = []
    var_23 = b'[tool.vulture]\nverbose = true\n'
    var_24 = module_0.make_config(var_22, var_16)
    var_25 = []
    var_26 = b'[tool.vulture]\nunknown_key = 1\n'
    var_27 = module_0.make_config(var_25, var_16)
    var_28 = []
    var_29 = b"[tool.vulture]\nmin_confidence = 'not_an_int'\n"
    var_30 = module_0.make_config(var_28, var_16)
    var_31 = '--min-confidence'
    var_32 = 'not_an_int'
    var_33 = [var_31, var_32]
    var_34 = None
    var_35 = module_0.make_config(var_33, var_34)
    var_36 = '-v'
    var_37 = [var_36]



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 25\n    paths = ["src/"]\n    exclude = ["tests/"]\n    ignore_names = ["unused_*"]\n    make_whitelist = true\n    sort_by_size = true\n    '
    var_9 = []
    var_10 = '75'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = 'path1.py'
    var_19 = 'path2.py'
    var_20 = 'dir/'
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--exclude'
    var_24 = '*.pyc,test_*.py'
    var_25 = 'src/'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--ignore-decorators'
    var_29 = '@app.route,@require_*'
    var_30 = '--ignore-names'
    var_31 = 'visit_*,do_*'
    var_32 = [var_28, var_29, var_30, var_31, var_25]
    var_33 = module_0.make_config(var_32)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35, var_25]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #15
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--verbose'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = b'\n    [tool.vulture]\n    paths = ["toml_path.py"]\n    min_confidence = 50\n    exclude = ["test*.py"]\n    '
    var_8 = []
    var_9 = 'cli_path.py'
    var_10 = [var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_14 = 'path.py'
    var_15 = [var_14]
    var_16 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = 'path.py'
    var_18 = [var_17]



# Parsed testcases at query #16
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    sort_by_size = true\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11]
    var_13 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 20\n    '
    var_14 = []
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = 'nonexistent.toml'
    var_18 = '--config'
    var_19 = 'test.py'
    var_20 = module_0.make_config(var_4)
    var_21 = '--help'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)



# Parsed testcases at query #17
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1.py'
    var_6 = 'path2.py'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = 'utf-8'
    var_11 = []
    var_12 = '20'
    var_13 = [var_2, var_12, var_4]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '--invalid-option'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = 'test.py'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '\n    [tool.vulture]\n    min_confidence = 5\n    '
    var_25 = []
    var_26 = '--config'
    var_27 = 'custom.toml'
    var_28 = [var_26, var_27, var_21]
    var_29 = module_0.make_config(var_28)
    var_30 = '--exclude'
    var_31 = 'a.py,b.py'
    var_32 = [var_30, var_31, var_21]
    var_33 = module_0.make_config(var_32)
    var_34 = '--ignore-decorators'
    var_35 = '@app.route,@require_*'
    var_36 = '--ignore-names'
    var_37 = 'visit_*,do_*'
    var_38 = [var_34, var_35, var_36, var_37, var_21]
    var_39 = module_0.make_config(var_38)
    var_40 = '--make-whitelist'
    var_41 = '--sort-by-size'
    var_42 = [var_40, var_41, var_21]
    var_43 = module_0.make_config(var_42)



# Parsed testcases at query #18
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    verbose = true\n    '
    var_9 = []
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    '
    var_11 = '80'
    var_12 = 'cli_path'
    var_13 = [var_2, var_11, var_12]
    var_14 = []
    var_15 = b'[tool.vulture]\nmin_confidence = 50\n'
    var_16 = module_0.make_config(var_14, var_3)
    var_17 = '\n    [tool.vulture]\n    invalid_key = "test"\n    paths = ["path"]\n    '
    var_18 = []
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_20 = []
    var_21 = '\n    [tool.vulture]\n    paths = ["path"]\n    verbose = true\n    '
    var_22 = []



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = 'path.py'
    var_9 = '--exclude'
    var_10 = 'test*.py,venv'
    var_11 = '--ignore-decorators'
    var_12 = 'app.route,require_*'
    var_13 = '--ignore-names'
    var_14 = 'visit_*,do_*'
    var_15 = '--make-whitelist'
    var_16 = '--sort-by-size'
    var_17 = '--verbose'
    var_18 = [var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1.py", "path2.py"]\n    '
    var_21 = []
    var_22 = 'override.py'
    var_23 = '100'
    var_24 = [var_22, var_4, var_23]
    var_25 = b"[tool.vulture]\npaths = ['test.py']\n"
    var_26 = []
    var_27 = 'pyproject.toml'
    var_28 = '[tool.vulture]\npaths = ["test.py"]\nmin_confidence = 20\n'
    var_29 = []
    var_30 = None
    var_31 = module_0.make_config(var_29, var_30)
    var_32 = '--config'
    var_33 = module_0.make_config(var_6)
    var_34 = []
    var_35 = None
    var_36 = module_0.make_config(var_34, var_35)
    var_37 = b"[tool.vulture]\nunknown_key = 'value'\n"
    var_38 = []
    var_39 = b"[tool.vulture]\npaths = ['test.py']\nmin_confidence = 'not_int'\n"
    var_40 = []
    var_41 = 'test.py'
    var_42 = '--min-confidence'
    var_43 = 'not_int'
    var_44 = [var_41, var_42, var_43]
    var_45 = module_0.make_config(var_44)
    var_46 = 'custom.toml'
    var_47 = '[tool.vulture]\npaths = ["test.py"]\n'
    var_48 = '--config'
    var_49 = module_0.make_config(var_43)
    var_50 = b"[tool.vulture]\npaths = ['test.py']\nverbose = true\n"
    var_51 = []



# Parsed testcases at query #21
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_11 = '80'
    var_12 = 'cli_path'
    var_13 = [var_2, var_11, var_12]
    var_14 = 'pyproject.toml'
    var_15 = b'\n            [tool.vulture]\n            min_confidence = 20\n            paths = ["detected_path"]\n            '
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = []
    var_19 = b'[]'
    var_20 = module_0.make_config(var_18, var_3)
    var_21 = []
    var_22 = b'\n        [tool.vulture]\n        unknown_key = "value"\n        paths = ["test"]\n        '
    var_23 = module_0.make_config(var_21, var_3)
    var_24 = []
    var_25 = b'\n        [tool.vulture]\n        min_confidence = "not_an_int"\n        paths = ["test"]\n        '
    var_26 = module_0.make_config(var_24, var_3)
    var_27 = '-v'
    var_28 = [var_27]
    var_29 = b'\n        [tool.vulture]\n        paths = ["test"]\n        '
    var_30 = module_0.make_config(var_28, var_26)



# Parsed testcases at query #22
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1.py'
    var_5 = 'path2.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    paths = ["src/"]\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_2, var_10]
    var_12 = '--exclude'
    var_13 = '*.pyc,venv'
    var_14 = '--ignore-decorators'
    var_15 = '@app.route,@login_required'
    var_16 = '--ignore-names'
    var_17 = 'helper_*'
    var_18 = '--make-whitelist'
    var_19 = '--sort-by-size'
    var_20 = '--verbose'
    var_21 = 'src/'
    var_22 = [var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--config'
    var_25 = 'nonexistent.toml'
    var_26 = 'path.py'
    var_27 = [var_24, var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--min-confidence'
    var_30 = '10'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = b'\n    [tool.vulture]\n    invalid_option = true\n    paths = ["test.py"]\n    '
    var_34 = []
    var_35 = b'\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["test.py"]\n    '
    var_36 = []
    var_37 = b'\n    [tool.vulture]\n    paths = []\n    '
    var_38 = []



# Parsed testcases at query #23
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["exclude1", "exclude2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    '
    var_12 = '80'
    var_13 = 'cli_path'
    var_14 = [var_2, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "test"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = b'[tool.vulture]\n'
    var_21 = module_0.make_config(var_19, var_3)



# Parsed testcases at query #24
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 75\n    verbose = true\n    '
    var_10 = []
    var_11 = 'cli_path'
    var_12 = '90'
    var_13 = [var_11, var_4, var_12]
    var_14 = '--exclude'
    var_15 = '*.py,test_*.py'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--ignore-names'
    var_23 = 'visit_*,do_*'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--make-whitelist'
    var_27 = '--sort-by-size'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = []
    var_31 = module_0.make_config(var_30)
    var_32 = '--config'
    var_33 = 'custom_config.toml'
    var_34 = [var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_37 = []
    var_38 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_39 = []



# Parsed testcases at query #25
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = b'\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["path3"]\n    exclude = ["test_*.py"]\n    '
    var_8 = '80'
    var_9 = [var_1, var_8]
    var_10 = '--exclude'
    var_11 = 'file1.py,dir/'
    var_12 = '--ignore-decorators'
    var_13 = '@app.route,@require_*'
    var_14 = '--ignore-names'
    var_15 = 'visit_*,do_*'
    var_16 = '--make-whitelist'
    var_17 = '--sort-by-size'
    var_18 = '--verbose'
    var_19 = '75'
    var_20 = '--config'
    var_21 = 'custom.toml'
    var_22 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_1, var_19, var_20, var_21, var_3, var_4]
    var_23 = module_0.make_config(var_22)
    var_24 = '--min-confidence'
    var_25 = '20'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = b'\n    [tool.vulture]\n    invalid_key = "test"\n    paths = ["test_path"]\n    '
    var_29 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test_path"]\n    '



# Parsed testcases at query #26
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = 'override.py'
    var_6 = '--min-confidence'
    var_7 = '20'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--exclude'
    var_10 = '*.pyc,__pycache__'
    var_11 = '--verbose'
    var_12 = [var_1, var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_16 = 'test.py'
    var_17 = '--min-confidence'
    var_18 = 'not_an_int'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '\n    [tool.vulture]\n    paths = ["test.py"]\n    verbose = true\n    '
    var_22 = '--config'
    var_23 = 'nonexistent.toml'
    var_24 = [var_16, var_22, var_23]
    var_25 = module_0.make_config(var_24)



# Parsed testcases at query #27
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 20\n    paths = ["dir1", "dir2"]\n    exclude = ["test_*.py"]\n    verbose = true\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #28
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n        [tool.vulture]\n        exclude = ["file*.py", "dir/"]\n        min_confidence = 10\n        paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = b'\n        [tool.vulture]\n        min_confidence = 10\n        paths = ["toml_path"]\n    '
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_4, var_13]
    var_15 = b'\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_16 = []
    var_17 = '--invalid-option'
    var_18 = 'value'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = b'\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_22 = []
    var_23 = []
    var_24 = module_0.make_config(var_23)
    var_25 = b'\n        [tool.vulture]\n        paths = ["some_path"]\n    '
    var_26 = []
    var_27 = 'path'
    var_28 = '--exclude'
    var_29 = 'file1.py,file2.py'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '--ignore-decorators'
    var_33 = '@app.route,@require_*'
    var_34 = [var_27, var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = '--ignore-names'
    var_37 = 'visit_*,do_*'
    var_38 = [var_27, var_36, var_37]
    var_39 = module_0.make_config(var_38)
    var_40 = '--make-whitelist'
    var_41 = [var_27, var_40]
    var_42 = module_0.make_config(var_41)
    var_43 = '--sort-by-size'
    var_44 = [var_27, var_43]
    var_45 = module_0.make_config(var_44)
    var_46 = b'\n        [tool.vulture]\n        paths = ["some_path"]\n        verbose = true\n    '
    var_47 = []
    var_48 = '\n            [tool.vulture]\n            paths = ["custom_path"]\n            min_confidence = 75\n        '
    var_49 = '--config'
    var_50 = module_0.make_config(var_24)
    var_51 = '--config'
    var_52 = 'nonexistent.toml'
    var_53 = 'some_path'
    var_54 = [var_51, var_52, var_53]
    var_55 = module_0.make_config(var_54)



# Parsed testcases at query #29
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = module_0.make_config(var_2, var_3)
    var_5 = b'\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_6 = 'test_config.toml'
    var_7 = []
    var_8 = 'cli_path'
    var_9 = '--min-confidence'
    var_10 = '20'
    var_11 = [var_8, var_9, var_10]
    var_12 = '--verbose'
    var_13 = [var_12]
    var_14 = b'\n[tool.vulture]\ninvalid_key = "value"\n'
    var_15 = 'invalid_config.toml'
    var_16 = []
    var_17 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = 'wrong_type.toml'
    var_19 = []
    var_20 = []
    var_21 = None
    var_22 = module_0.make_config(var_20, var_21)
    var_23 = 'path'
    var_24 = '--config'
    var_25 = 'nonexistent.toml'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)



# Parsed testcases at query #30
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'src'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = false\n    '
    var_11 = '90'
    var_12 = [var_3, var_11, var_2]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["src"]\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["src"]\n    '
    var_18 = []
    var_19 = '\n        [tool.vulture]\n        min_confidence = 42\n        paths = ["some_path"]\n        '
    var_20 = '--config'
    var_21 = module_0.make_config(var_14)
    var_22 = '--exclude'
    var_23 = 'a.py,b.py'
    var_24 = '--ignore-names'
    var_25 = 'foo,bar'
    var_26 = [var_22, var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1"]\n    verbose = true\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = 10\n    '
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_15 = []
    var_16 = '--min-confidence'
    var_17 = '10'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '\n    [tool.vulture]\n    paths = ["some_path"]\n    '
    var_21 = []
    var_22 = '--exclude'
    var_23 = 'test_*.py,docs'
    var_24 = '--ignore-decorators'
    var_25 = '@app.route'
    var_26 = '--ignore-names'
    var_27 = 'private_*'
    var_28 = 'main.py'
    var_29 = [var_22, var_23, var_24, var_25, var_26, var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--make-whitelist'
    var_32 = '--sort-by-size'
    var_33 = 'file.py'
    var_34 = [var_31, var_32, var_33]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #32
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_5, var_13]
    var_15 = 'pyproject.toml'
    var_16 = '--config'
    var_17 = []
    var_18 = ''
    var_19 = module_0.make_config(var_17, var_3)
    var_20 = '\n        [tool.vulture]\n        invalid_key = "value"\n        paths = ["path"]\n        '
    var_21 = []
    var_22 = module_0.make_config(var_21, var_18)
    var_23 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n        paths = ["path"]\n        '
    var_24 = []
    var_25 = module_0.make_config(var_24, var_18)
    var_26 = '--min-confidence'
    var_27 = 'not_an_int'
    var_28 = [var_26, var_27]
    var_29 = None
    var_30 = module_0.make_config(var_28, var_29)



# Parsed testcases at query #33
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '90'
    var_12 = 'custom_path'
    var_13 = [var_4, var_11, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = 'some_path'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #34
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 20\n    exclude = ["file1.py", "file2.py"]\n    paths = ["src"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = '\n    [tool.vulture]\n    unknown_key = 5\n    paths = ["src"]\n    '
    var_15 = []
    var_16 = module_0.make_config(var_15, var_13)
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["src"]\n    '
    var_18 = []
    var_19 = module_0.make_config(var_18, var_13)



# Parsed testcases at query #35
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 30\n    exclude = ["exclude1", "exclude2"]\n    '
    var_10 = 'utf-8'
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '80'
    var_14 = [var_12, var_4, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_18 = 'test_path'
    var_19 = [var_18]
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_21 = 'test_path'
    var_22 = [var_21]
    var_23 = '--version'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--help'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #36
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 80\n    exclude = ["test_*.py", "docs"]\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    min_confidence = 80\n    paths = ["src"]\n    '
    var_12 = '30'
    var_13 = 'cli_path'
    var_14 = [var_2, var_12, var_13]
    var_15 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = '--min-confidence'
    var_20 = '50'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--config'
    var_24 = 'nonexistent.toml'
    var_25 = 'some_path'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = b'\n        [tool.vulture]\n        min_confidence = 90\n        paths = ["test_path"]\n        '
    var_29 = '--config'
    var_30 = module_0.make_config(var_20)
    var_31 = '--exclude'
    var_32 = 'file1.py,file2.py'
    var_33 = '--ignore-decorators'
    var_34 = 'deco1,deco2'
    var_35 = '--ignore-names'
    var_36 = 'name1,name2'
    var_37 = '--make-whitelist'
    var_38 = '--sort-by-size'
    var_39 = '75'
    var_40 = [var_31, var_32, var_33, var_34, var_35, var_36, var_37, var_38, var_22, var_20, var_39, var_5, var_6]
    var_41 = module_0.make_config(var_40)



# Parsed testcases at query #37
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["test_exclude"]\n    min_confidence = 50\n    paths = ["test_path"]\n    '
    var_1 = '--min-confidence'
    var_2 = '80'
    var_3 = [var_1, var_2]
    var_4 = '30'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_1, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '--verbose'
    var_10 = '--make-whitelist'
    var_11 = 'my_path'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = [var_11]
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'a.py,b.py,c.py'
    var_18 = 'path'
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--ignore-decorators'
    var_22 = '@app.route,@require_*'
    var_23 = [var_21, var_22, var_18]
    var_24 = module_0.make_config(var_23)
    var_25 = '--ignore-names'
    var_26 = 'visit_*,do_*'
    var_27 = [var_25, var_26, var_18]
    var_28 = module_0.make_config(var_27)
    var_29 = '--sort-by-size'
    var_30 = [var_29, var_18]
    var_31 = module_0.make_config(var_30)
    var_32 = '--config'
    var_33 = 'nonexistent.toml'
    var_34 = [var_32, var_33, var_18]
    var_35 = module_0.make_config(var_34)
    var_36 = 'cli_path'
    var_37 = [var_36]
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '--help'
    var_42 = [var_41]
    var_43 = module_0.make_config(var_42)
    var_44 = []
    var_45 = module_0.make_config(var_44)



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    min_confidence = 25\n    exclude = ["test_*.py", "venv"]\n    paths = ["src"]\n    '
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 25\n    verbose = false\n    '
    var_10 = '75'
    var_11 = [var_1, var_10, var_3]
    var_12 = '--make-whitelist'
    var_13 = '--sort-by-size'
    var_14 = [var_12, var_13, var_4]
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'a.py,b.py'
    var_18 = '--ignore-decorators'
    var_19 = 'deco1,deco2'
    var_20 = '--ignore-names'
    var_21 = 'name1,name2'
    var_22 = 'path'
    var_23 = [var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = b'[tool.vulture]\nmin_confidence = 80\n'
    var_26 = '--config'
    var_27 = module_0.make_config(var_2)
    var_28 = '--min-confidence'
    var_29 = '10'
    var_30 = [var_28, var_29]
    var_31 = module_0.make_config(var_30)



# Parsed testcases at query #39
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '-v'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'test.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test_*.py", "venv"]\n    min_confidence = 20\n    sort_by_size = true\n    paths = ["src", "main.py"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_3, var_10]
    var_12 = '--min-confidence'
    var_13 = '10'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["test.py"]\n    '
    var_19 = []
    var_20 = '\n        [tool.vulture]\n        verbose = true\n        paths = ["test.py"]\n        '
    var_21 = '--config'



# Parsed testcases at query #40
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 75\n    exclude = ["exclude1", "exclude2"]\n    '
    var_9 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    min_confidence = 10\n    '
    var_10 = 'cli_path'
    var_11 = '90'
    var_12 = [var_10, var_3, var_11]
    var_13 = '--exclude'
    var_14 = 'a.py,b.py'
    var_15 = '--ignore-decorators'
    var_16 = 'dec1,dec2'
    var_17 = '--ignore-names'
    var_18 = 'name1,name2'
    var_19 = [var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--make-whitelist'
    var_22 = '--sort-by-size'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--config'
    var_26 = 'custom.toml'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = []
    var_30 = module_0.make_config(var_29)
    var_31 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_32 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '



# Parsed testcases at query #41
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    min_confidence = 50\n    paths = ["src/"]\n    exclude = ["test_*.py"]\n    '
    var_1 = '--min-confidence'
    var_2 = '75'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = '--verbose'
    var_7 = '--sort-by-size'
    var_8 = 'myfile.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = '[tool.vulture]\nmin_confidence = 30\npaths = ["dir/"]\n'
    var_12 = module_0.make_config()
    var_13 = 'custom_config.toml'
    var_14 = '[tool.vulture]\nmin_confidence = 40\npaths = ["src/"]\n'
    var_15 = '--config'
    var_16 = module_0.make_config(var_3)
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '[tool.vulture]\ninvalid_key = 5\npaths = ["x"]\n'
    var_20 = '[tool.vulture]\nmin_confidence = "not_int"\npaths = ["x"]\n'
    var_21 = '[tool.vulture]\nmin_confidence = 10\n'



# Parsed testcases at query #42
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
    var_10 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 30\n    ignore_names = ["test_*"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = '80'
    var_14 = [var_12, var_6, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    paths = "not_a_list"\n    '
    var_21 = []
    var_22 = '\n    [tool.vulture]\n    paths = ["path"]\n    exclude = ["file1.py,file2.py"]\n    '
    var_23 = []
    var_24 = '--verbose'
    var_25 = [var_24]



# Parsed testcases at query #43
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = '30'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_1, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '



# Parsed testcases at query #44
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '80'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_10 = []
    var_11 = 'override.py'
    var_12 = '50'
    var_13 = [var_11, var_4, var_12]
    var_14 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = 'test.py'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--version'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--help'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #45
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_1 = '--min-confidence'
    var_2 = '20'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = '--exclude'
    var_8 = 'file*.py,dir/'
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = module_0.make_config(var_9)
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '



# Parsed testcases at query #46
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = '75'
    var_11 = [var_2, var_10]
    var_12 = "[tool.vulture]\ninvalid_key = true\npaths = ['test']"
    var_13 = "[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test']"
    var_14 = []
    var_15 = module_0.make_config(var_14)



# Parsed testcases at query #47
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    paths = ["src", "tests"]\n    min_confidence = 80\n    exclude = ["*.pyc", "venv"]\n    '
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    paths = ["src"]\n    min_confidence = 80\n    '
    var_12 = 'custom_path'
    var_13 = '30'
    var_14 = [var_12, var_4, var_13]
    var_15 = []
    var_16 = b'[tool.vulture]\n'
    var_17 = module_0.make_config(var_15, var_3)
    var_18 = '--version'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["src"]\n    '
    var_22 = []
    var_23 = module_0.make_config(var_22, var_19)
    var_24 = b'\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["src"]\n    '
    var_25 = []
    var_26 = module_0.make_config(var_25, var_19)



# Parsed testcases at query #48
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--sort-by-size'
    var_5 = '--verbose'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '90'
    var_13 = 'cli_path'
    var_14 = [var_2, var_12, var_5, var_13]
    var_15 = '--min-confidence'
    var_16 = '10'
    var_17 = [var_15, var_16]
    var_18 = b'[tool.vulture]\n'
    var_19 = module_0.make_config(var_17, var_5)
    var_20 = b'[tool.vulture]\nunknown_key = true\n'
    var_21 = []



# Parsed testcases at query #49
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nexclude = ["exclude1", "exclude2"]\nmin_confidence = 75\npaths = ["path_a", "path_b"]\nverbose = true\n'
    var_7 = 'test_config.toml'
    var_8 = 'cli_path'
    var_9 = '--min-confidence'
    var_10 = '90'
    var_11 = [var_8, var_9, var_10]
    var_12 = '--min-confidence'
    var_13 = '10'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--verbose'
    var_17 = [var_16]
    var_18 = 'pyproject.toml'
    var_19 = '\n[tool.vulture]\nmin_confidence = 30\npaths = ["auto_detected_path"]\n'
    var_20 = '--config'
    var_21 = 'test_path'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)



# Parsed testcases at query #50
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = b'\n    [tool.vulture]\n    exclude = ["test_*.py", "docs/"]\n    ignore_decorators = ["decorator1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 50\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = []
    var_6 = '--min-confidence'
    var_7 = '80'
    var_8 = '--verbose'
    var_9 = 'cli_path.py'
    var_10 = [var_6, var_7, var_8, var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_14 = []
    var_15 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_16 = []
    var_17 = 'pyproject.toml'
    var_18 = b'\n            [tool.vulture]\n            min_confidence = 30\n            paths = ["test.py"]\n            '
    var_19 = '--config'
    var_20 = module_0.make_config(var_2)
    var_21 = '--verbose'
    var_22 = [var_21]



# Parsed testcases at query #51
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    min_confidence = 50\n    paths = ["src"]\n    '
    var_10 = []
    var_11 = '90'
    var_12 = [var_2, var_11]
    var_13 = '--exclude'
    var_14 = 'file1.py,dir/'
    var_15 = '--ignore-decorators'
    var_16 = 'deco1,deco2'
    var_17 = '--ignore-names'
    var_18 = 'name1,name2'
    var_19 = '--make-whitelist'
    var_20 = '--sort-by-size'
    var_21 = '--config'
    var_22 = 'custom.toml'
    var_23 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_5, var_6]
    var_24 = module_0.make_config(var_23)
    var_25 = []
    var_26 = module_0.make_config(var_25)
    var_27 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["src"]\n    '
    var_28 = []
    var_29 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["src"]\n    '
    var_30 = []



# Parsed testcases at query #52
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'test.py'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 10\n    paths = ["path1", "path2"]\n    '
    var_8 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1"]\n    '
    var_9 = '80'
    var_10 = 'path2'
    var_11 = [var_2, var_9, var_10]
    var_12 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path1"]\n    '
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = '--verbose'
    var_17 = '--make-whitelist'
    var_18 = [var_16, var_17, var_4]
    var_19 = module_0.make_config(var_18)
    var_20 = '--exclude'
    var_21 = 'file1.py,file2.py'
    var_22 = [var_20, var_21, var_4]
    var_23 = module_0.make_config(var_22)
    var_24 = [var_4]
    var_25 = None
    var_26 = module_0.make_config(var_24, var_25)
    var_27 = '\n    [tool.vulture]\n    paths = ["path1"]\n    '
    var_28 = '--verbose'
    var_29 = [var_28]
    var_30 = '--sort-by-size'
    var_31 = [var_30, var_4]
    var_32 = module_0.make_config(var_31)
    var_33 = '--ignore-names'
    var_34 = 'name1,name2'
    var_35 = [var_33, var_34, var_4]
    var_36 = module_0.make_config(var_35)
    var_37 = '--ignore-decorators'
    var_38 = 'deco1,deco2'
    var_39 = [var_37, var_38, var_4]
    var_40 = module_0.make_config(var_39)
    var_41 = [var_4]
    var_42 = module_0.make_config(var_41)



# Parsed testcases at query #53
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = b'\n    [tool.vulture]\n    exclude = ["file1.py", "file2.py"]\n    min_confidence = 20\n    verbose = true\n    '
    var_11 = []
    var_12 = '80'
    var_13 = [var_5, var_12]
    var_14 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = []
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)



# Parsed testcases at query #54
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = true\n    paths = ["toml_path"]\n    '
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_2, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--invalid-option'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--version'
    var_28 = [var_27]
    var_29 = module_0.make_config(var_28)



# Parsed testcases at query #55
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1.py'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    paths = ["src/"]\n    '
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    min_confidence = 75\n    paths = ["src/"]\n    '
    var_12 = '90'
    var_13 = [var_4, var_12]
    var_14 = b'\n    [tool.vulture]\n    min_confidence = 75\n    '
    var_15 = []
    var_16 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["src/"]\n    '
    var_17 = []
    var_18 = b'\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["src/"]\n    '
    var_19 = []
    var_20 = 'path.py'
    var_21 = '--exclude'
    var_22 = ''
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23, var_1)
    var_25 = 'path2.py'
    var_26 = 'dir/'
    var_27 = [var_3, var_25, var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = '--make-whitelist'
    var_30 = '--sort-by-size'
    var_31 = [var_20, var_29, var_30]
    var_32 = module_0.make_config(var_31, var_1)



# Parsed testcases at query #56
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_3 = []
    var_4 = '\n    [tool.vulture]\n    min_confidence = 10\n    verbose = true\n    paths = ["path1"]\n    '
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = 'path2'
    var_8 = 'path3'
    var_9 = [var_5, var_6, var_7, var_8]
    var_10 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_11 = []
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = '--min-confidence'
    var_15 = 'invalid'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--unknown-arg'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--version'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["path1"]\n    '
    var_28 = []
    var_29 = '\n        [tool.vulture]\n        min_confidence = 20\n        paths = ["test_path"]\n        '
    var_30 = '--config'
    var_31 = 'nonexistent.toml'
    var_32 = 'path1'
    var_33 = [var_30, var_31, var_32]
    var_34 = module_0.make_config(var_33)



# Parsed testcases at query #57
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 50\npaths = ["toml_path1"]\nsort_by_size = true\n'
    var_7 = 'cli_path'
    var_8 = '--min-confidence'
    var_9 = '90'
    var_10 = [var_7, var_8, var_9]
    var_11 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_12 = []
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '\n[tool.vulture]\nmin_confidence = 30\npaths = ["default_path"]\n'
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nverbose = true\npaths = ["test_path"]\n'
    var_19 = []



# Parsed testcases at query #58
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = []
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = 'path1.py'
    var_7 = 'path2.py'
    var_8 = [var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '90'
    var_13 = 'cli_path.py'
    var_14 = [var_4, var_12, var_13]
    var_15 = 'pyproject.toml'
    var_16 = var_2 / var_15
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '--min-confidence'
    var_21 = '50'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_25 = []
    var_26 = b'\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["test.py"]\n    '
    var_27 = []
    var_28 = '--min-confidence'
    var_29 = 'not_a_number'
    var_30 = 'test.py'
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = []



# Parsed testcases at query #59
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    ignore_decorators = ["deco1"]\n    min_confidence = 75\n    paths = ["src"]\n    '
    var_10 = []
    var_11 = '100'
    var_12 = [var_2, var_11]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'[tool.vulture]\nunknown_key = true\n'
    var_16 = []
    var_17 = b"[tool.vulture]\nmin_confidence = '50'\n"
    var_18 = []
    var_19 = '--exclude'
    var_20 = 'test.py'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = b'[tool.vulture]\nverbose = true\n'
    var_24 = []



# Parsed testcases at query #60
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = 'utf-8'
    var_6 = []
    var_7 = '--min-confidence'
    var_8 = '20'
    var_9 = '--verbose'
    var_10 = [var_7, var_8, var_9]
    var_11 = '15'
    var_12 = '--sort-by-size'
    var_13 = '--make-whitelist'
    var_14 = '--exclude'
    var_15 = 'test.py,test2.py'
    var_16 = '--ignore-decorators'
    var_17 = 'deco1'
    var_18 = '--ignore-names'
    var_19 = 'name1'
    var_20 = [var_7, var_11, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_0]
    var_21 = module_0.make_config(var_20)
    var_22 = []
    var_23 = module_0.make_config(var_22)
    var_24 = b'[tool.vulture]\ninvalid_key = 1\n'
    var_25 = 'path1'
    var_26 = [var_25]
    var_27 = b"[tool.vulture]\nmin_confidence = 'invalid'\n"
    var_28 = 'path1'
    var_29 = [var_28]



# Parsed testcases at query #61
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '80'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["toml_path1.py", "toml_path2.py"]\n'
    var_10 = []
    var_11 = 'cli_path.py'
    var_12 = '50'
    var_13 = [var_11, var_4, var_12]
    var_14 = '[tool.vulture]\ninvalid_key = true\n'
    var_15 = []
    var_16 = "[tool.vulture]\nmin_confidence = 'not_an_int'\n"
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = 'some_path.py'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = 'path.py'
    var_24 = '--make-whitelist'
    var_25 = '--sort-by-size'
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.make_config(var_26)
    var_28 = '--exclude'
    var_29 = 'a.py,b.py'
    var_30 = '--ignore-decorators'
    var_31 = 'dec1,dec2'
    var_32 = '--ignore-names'
    var_33 = 'name1,name2'
    var_34 = [var_23, var_28, var_29, var_30, var_31, var_32, var_33]
    var_35 = module_0.make_config(var_34)



# Parsed testcases at query #62
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    exclude = ["file*.py"]\n    ignore_decorators = ["deco1"]\n    min_confidence = 50\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = '90'
    var_12 = [var_2, var_11]
    var_13 = '--min-confidence'
    var_14 = '80'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = b'\n    [tool.vulture]\n    unknown_key = true\n    paths = ["test"]\n    '
    var_18 = []
    var_19 = module_0.make_config(var_18, var_14)
    var_20 = b'\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["test"]\n    '
    var_21 = []
    var_22 = module_0.make_config(var_21, var_14)
    var_23 = '--verbose'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24, var_22)



# Parsed testcases at query #63
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 80\n    paths = ["src/"]\n    verbose = true\n    '
    var_10 = []
    var_11 = '30'
    var_12 = [var_2, var_11]
    var_13 = '--exclude'
    var_14 = 'test*.py,venv'
    var_15 = '--ignore-names'
    var_16 = 'foo,bar'
    var_17 = '--sort-by-size'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--make-whitelist'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_24 = []
    var_25 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_26 = []
    var_27 = []
    var_28 = b'[]'
    var_29 = module_0.make_config(var_27, var_3)



# Parsed testcases at query #64
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = 'custom.py'
    var_12 = '80'
    var_13 = [var_11, var_4, var_12]
    var_14 = '--config'
    var_15 = 'nonexistent.toml'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--config'
    var_19 = 'nonexistent.toml'
    var_20 = '--min-confidence'
    var_21 = 'invalid'
    var_22 = [var_18, var_19, var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test.py"]\n    '
    var_25 = []
    var_26 = '\n    [tool.vulture]\n    min_confidence = "not an int"\n    paths = ["test.py"]\n    '
    var_27 = []
    var_28 = '--version'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--help'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)



# Parsed testcases at query #65
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 20\n    paths = ["toml_path1"]\n    verbose = true\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_3, var_11]
    var_13 = b'\n    [tool.vulture]\n    paths = ["test_path"]\n    '
    var_14 = []
    var_15 = '--min-confidence'
    var_16 = '50'
    var_17 = [var_15, var_16]
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["test"]\n    '
    var_21 = []
    var_22 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test"]\n    '
    var_23 = []



# Parsed testcases at query #66
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    paths = ["src"]\n    min_confidence = 50\n    exclude = ["test_*.py"]\n    ignore_decorators = ["@app.route"]\n    ignore_names = ["private_*"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    '
    var_1 = []
    var_2 = '--min-confidence'
    var_3 = '80'
    var_4 = 'src'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'test_file.py'
    var_7 = [var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '--exclude'
    var_10 = 'test_*.py,docs'
    var_11 = '--sort-by-size'
    var_12 = [var_9, var_10, var_11, var_4]
    var_13 = module_0.make_config(var_12)
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = '\n    [tool.vulture]\n    paths = ["src"]\n    invalid_key = "value"\n    '
    var_17 = []



# Parsed testcases at query #67
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 20\n    exclude = ["file1.py", "file2.py"]\n    paths = ["src"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = b'[tool.vulture]\ninvalid_key = true\n'
    var_13 = []
    var_14 = b'[tool.vulture]\nmin_confidence = "high"\n'
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = 'test_file.py'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--exclude'
    var_22 = 'file1.py,file2.py'
    var_23 = [var_21, var_22, var_18]
    var_24 = module_0.make_config(var_23)
    var_25 = '--ignore-decorators'
    var_26 = '@app.route,@require'
    var_27 = [var_25, var_26, var_18]
    var_28 = module_0.make_config(var_27)
    var_29 = '--ignore-names'
    var_30 = 'visit_*,do_*'
    var_31 = [var_29, var_30, var_18]
    var_32 = module_0.make_config(var_31)
    var_33 = '--make-whitelist'
    var_34 = [var_33, var_18]
    var_35 = module_0.make_config(var_34)
    var_36 = '--sort-by-size'
    var_37 = [var_36, var_18]
    var_38 = module_0.make_config(var_37)
    var_39 = '--verbose'
    var_40 = [var_39, var_18]
    var_41 = module_0.make_config(var_40)
    var_42 = b'[tool.vulture]\nmin_confidence = 30\n'
    var_43 = '--config'
    var_44 = 'test_file.py'
    var_45 = module_0.make_config(var_3)



# Parsed testcases at query #68
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 30\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    sort_by_size = true\n    verbose = true\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    min_confidence = 30\n    '
    var_12 = 'cli_path'
    var_13 = '80'
    var_14 = [var_12, var_4, var_13]
    var_15 = '\n    [tool.vulture]\n    paths = ["test_path"]\n    '
    var_16 = []
    var_17 = '--min-confidence'
    var_18 = '10'
    var_19 = [var_17, var_18]
    var_20 = b''
    var_21 = module_0.make_config(var_19, var_5)



# Parsed testcases at query #69
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test*.py", "docs"]\npaths = ["src"]\n'
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = '\n[tool.vulture]\nmin_confidence = 40\npaths = ["src"]\n'
    var_13 = '--config'
    var_14 = module_0.make_config(var_2)
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n[tool.vulture]\nmin_confidence = 20\n'
    var_18 = []
    var_19 = '--verbose'
    var_20 = [var_19]
    var_21 = '--sort-by-size'
    var_22 = 'src'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #70
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = b'\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    min_confidence = 10\n    '
    var_4 = 'pyproject.toml'
    var_5 = []
    var_6 = '--min-confidence'
    var_7 = '20'
    var_8 = 'cli_path'
    var_9 = [var_6, var_7, var_8]
    var_10 = b'\n    [tool.vulture]\n    paths = ["default_path"]\n    verbose = true\n    '
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = '--verbose'
    var_14 = 'verbose_path'
    var_15 = [var_13, var_14]
    var_16 = 'only_path'
    var_17 = [var_16]
    var_18 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path"]\n    '
    var_19 = 'bad_pyproject.toml'
    var_20 = []
    var_21 = b'\n    [tool.vulture]\n    paths = "not_a_list"\n    '
    var_22 = 'bad_type.toml'
    var_23 = []
    var_24 = 'custom_config.toml'
    var_25 = b'\n    [tool.vulture]\n    paths = ["custom_path"]\n    '
    var_26 = '--config'
    var_27 = 'nonexistent.toml'
    var_28 = 'fallback_path'



# Parsed testcases at query #71
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    min_confidence = 10\n    sort_by_size = true\n    paths = ["path3", "path4"]\n    '
    var_10 = []
    var_11 = 'custom_path'
    var_12 = '75'
    var_13 = [var_11, var_4, var_12]
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["path"]\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_19 = []



# Parsed testcases at query #72
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_4, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '\n    [tool.vulture]\n    paths = ["path1"]\n    '
    var_22 = []



# Parsed testcases at query #73
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_2, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '--config'
    var_16 = 'nonexistent.toml'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n                [tool.vulture]\n                min_confidence = 30\n                '
    var_20 = 'test.py'
    var_21 = [var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #74
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test_file.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test*.py"]\n    min_confidence = 30\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = []
    var_15 = b'[tool.vulture]\\n'
    var_16 = module_0.make_config(var_14, var_4)
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []



# Parsed testcases at query #75
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["test*.py", "docs"]\n    ignore_decorators = ["@app.route"]\n    ignore_names = ["private_*"]\n    make_whitelist = true\n    min_confidence = 75\n    sort_by_size = true\n    verbose = false\n    paths = ["src", "tests"]\n    '
    var_11 = []
    var_12 = '90'
    var_13 = [var_5, var_12, var_7]
    var_14 = []
    var_15 = b''
    var_16 = module_0.make_config(var_14, var_3)
    var_17 = b'\n    [tool.vulture]\n    unknown_key = true\n    '
    var_18 = 'path'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19, var_3)
    var_21 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_22 = 'path'
    var_23 = [var_22]
    var_24 = module_0.make_config(var_23, var_3)
    var_25 = '--version'
    var_26 = [var_25]
    var_27 = None
    var_28 = module_0.make_config(var_26, var_27)
    var_29 = '--help'
    var_30 = [var_29]
    var_31 = None
    var_32 = module_0.make_config(var_30, var_31)



# Parsed testcases at query #76
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7, var_1)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test*.py"]\n    verbose = true\n    '
    var_10 = []
    var_11 = '70'
    var_12 = [var_5, var_11]
    var_13 = '\n    [tool.vulture]\n    paths = ["src", "tests"]\n    '
    var_14 = []
    var_15 = []
    var_16 = ''
    var_17 = module_0.make_config(var_15, var_3)
    var_18 = '\n    [tool.vulture]\n    invalid_key = true\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_21 = []



# Parsed testcases at query #77
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.make_config(var_1)
    var_3 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_4 = []
    var_5 = '--min-confidence'
    var_6 = '20'
    var_7 = 'cli_file.py'
    var_8 = [var_5, var_6, var_7]
    var_9 = '--exclude'
    var_10 = 'file1.py,file2.py'
    var_11 = 'test.py'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = '--ignore-decorators'
    var_15 = '@app.route,@require_*'
    var_16 = [var_14, var_15, var_11]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-names'
    var_19 = 'visit_*,do_*'
    var_20 = [var_18, var_19, var_11]
    var_21 = module_0.make_config(var_20)
    var_22 = '--make-whitelist'
    var_23 = [var_22, var_11]
    var_24 = module_0.make_config(var_23)
    var_25 = '--sort-by-size'
    var_26 = [var_25, var_11]
    var_27 = module_0.make_config(var_26)
    var_28 = '--verbose'
    var_29 = [var_28, var_11]
    var_30 = module_0.make_config(var_29)
    var_31 = '--config'
    var_32 = 'nonexistent.toml'
    var_33 = [var_31, var_32, var_11]
    var_34 = module_0.make_config(var_33)
    var_35 = []
    var_36 = module_0.make_config(var_35)
    var_37 = '\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_38 = []
    var_39 = '\n    [tool.vulture]\n    min_confidence = "ten"\n    paths = ["test.py"]\n    '
    var_40 = []



# Parsed testcases at query #78
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '\n    [tool.vulture]\n    '
    var_1 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_2 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1"]\n    '
    var_3 = '--min-confidence'
    var_4 = '20'
    var_5 = 'path2'
    var_6 = [var_3, var_4, var_5]
    var_7 = '30'
    var_8 = '--verbose'
    var_9 = 'path3'
    var_10 = [var_3, var_7, var_8, var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = '\n    [tool.vulture]\n    '
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_14 = '\n    [tool.vulture]\n    min_confidence = "invalid"\n    paths = ["path1"]\n    '
    var_15 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["path1"]\n    '



# Parsed testcases at query #79
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    paths = ["src"]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    '
    var_10 = []
    var_11 = 'custom_path'
    var_12 = '80'
    var_13 = [var_11, var_4, var_12]
    var_14 = []
    var_15 = '[tool.vulture]\nmin_confidence = 10\n'
    var_16 = module_0.make_config(var_14, var_3)
    var_17 = []
    var_18 = "[tool.vulture]\nmin_confidence = 'invalid'\n"
    var_19 = module_0.make_config(var_17, var_3)
    var_20 = []
    var_21 = "[tool.vulture]\nunknown_key = 10\npaths = ['test']\n"
    var_22 = module_0.make_config(var_20, var_3)
    var_23 = '--sort-by-size'
    var_24 = '--make-whitelist'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--exclude'
    var_28 = 'file1.py,file2.py'
    var_29 = 'path'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = '--ignore-decorators'
    var_33 = '@app.route,@require_*'
    var_34 = '--ignore-names'
    var_35 = 'visit_*,do_*'
    var_36 = [var_32, var_33, var_34, var_35, var_29]
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #80
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1.py'
    var_5 = 'path2.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["src"]\n    exclude = ["test_*.py"]\n    verbose = true\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = true\n    '
    var_13 = []
    var_14 = '\n    [tool.vulture]\n    min_confidence = "high"\n    '
    var_15 = []
    var_16 = []
    var_17 = ''
    var_18 = module_0.make_config(var_16, var_3)
    var_19 = 'somefile.py'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--exclude'
    var_23 = 'test_*.py,docs'
    var_24 = '--ignore-decorators'
    var_25 = '@app.route'
    var_26 = '--ignore-names'
    var_27 = 'private_*'
    var_28 = 'file.py'
    var_29 = [var_22, var_23, var_24, var_25, var_26, var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--make-whitelist'
    var_32 = '--sort-by-size'
    var_33 = '--verbose'
    var_34 = [var_31, var_32, var_33, var_28]
    var_35 = module_0.make_config(var_34)
    var_36 = '--config'
    var_37 = 'nonexistent.toml'
    var_38 = [var_36, var_37, var_28]
    var_39 = module_0.make_config(var_38)



# Parsed testcases at query #81
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 30\n    paths = ["src"]\n    '
    var_10 = 'utf-8'
    var_11 = []
    var_12 = '80'
    var_13 = [var_4, var_12]
    var_14 = '--exclude'
    var_15 = 'file1.py,file2.py'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--ignore-decorators'
    var_19 = '@app.route,@require_*'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--ignore-names'
    var_23 = 'visit_*,do_*'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--make-whitelist'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--sort-by-size'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30)
    var_32 = []
    var_33 = module_0.make_config(var_32)
    var_34 = '\n    [tool.vulture]\n    invalid_key = 10\n    paths = ["src"]\n    '
    var_35 = []
    var_36 = '\n    [tool.vulture]\n    min_confidence = "high"\n    paths = ["src"]\n    '
    var_37 = []
    var_38 = "[tool.vulture]\nmin_confidence = 42\npaths = ['test_path']\n"
    var_39 = '--config'
    var_40 = module_0.make_config(var_33)
    var_41 = [var_6]



# Parsed testcases at query #82
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py"]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = '--make-whitelist'
    var_13 = 'path'
    var_14 = [var_12, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #83
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '-v'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    min_confidence = 20\n    exclude = ["test_*.py"]\n    paths = ["src"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_4, var_11]
    var_13 = []
    var_14 = b''
    var_15 = module_0.make_config(var_13, var_3)
    var_16 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_17 = []
    var_18 = module_0.make_config(var_17, var_14)
    var_19 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_20 = []
    var_21 = module_0.make_config(var_20, var_14)



# Parsed testcases at query #84
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = 'test.toml'
    var_10 = []
    var_11 = b'\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["path1"]\n    '
    var_12 = 'precedence.toml'
    var_13 = '--min-confidence'
    var_14 = '80'
    var_15 = 'cli_path'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'pyproject.toml'
    var_18 = '\n    [tool.vulture]\n    min_confidence = 20\n    paths = ["auto_path"]\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--min-confidence'
    var_22 = '30'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = b'\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_26 = 'bad.toml'
    var_27 = []



# Parsed testcases at query #85
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_5 = []
    var_6 = 'cli_path'
    var_7 = '--min-confidence'
    var_8 = '20'
    var_9 = [var_6, var_7, var_8]
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = "[tool.vulture]\ninvalid_key = true\npaths = ['test.py']"
    var_13 = []
    var_14 = "[tool.vulture]\nmin_confidence = 'not_an_int'\npaths = ['test.py']"
    var_15 = []
    var_16 = '--version'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--help'
    var_20 = [var_19]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #86
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = 'path1'
    var_2 = 'path2'
    var_3 = '--verbose'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = '80'
    var_10 = [var_4, var_9]
    var_11 = []
    var_12 = module_0.make_config(var_11)
    var_13 = b'\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_14 = module_0.make_config(tomlfile=var_11)
    var_15 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_16 = module_0.make_config(tomlfile=var_11)
    var_17 = 'pyproject.toml'
    var_18 = '--config'
    var_19 = module_0.make_config(var_3)



# Parsed testcases at query #87
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = b'\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1"]\n    min_confidence = 25\n    '
    var_9 = []
    var_10 = '80'
    var_11 = [var_2, var_10]
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = b'\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = b'\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = '--verbose'
    var_19 = [var_18]
    var_20 = '--exclude'
    var_21 = 'file1.py,file2.py'
    var_22 = 'path'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--config'
    var_26 = 'custom.toml'
    var_27 = [var_25, var_26, var_22]
    var_28 = module_0.make_config(var_27)
    var_29 = '--make-whitelist'
    var_30 = '--sort-by-size'
    var_31 = [var_29, var_30, var_22]
    var_32 = module_0.make_config(var_31)



# Parsed testcases at query #88
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = 'cli_path.py'
    var_12 = '80'
    var_13 = [var_11, var_4, var_12]
    var_14 = '--min-confidence'
    var_15 = '50'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path.py"]\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    paths = ["path.py"]\n    min_confidence = "not_an_int"\n    '
    var_21 = []



# Parsed testcases at query #89
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["test*.py", "temp/"]\n    ignore_decorators = ["deco1"]\n    ignore_names = ["name1"]\n    make_whitelist = true\n    min_confidence = 75\n    sort_by_size = true\n    verbose = true\n    paths = ["src/"]\n    '
    var_9 = []
    var_10 = '90'
    var_11 = '--verbose'
    var_12 = [var_2, var_10, var_11]
    var_13 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = []
    var_18 = ''
    var_19 = module_0.make_config(var_17, var_3)
    var_20 = '--config'
    var_21 = module_0.make_config(var_18)
    var_22 = '--config'
    var_23 = 'nonexistent.toml'
    var_24 = 'path'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--exclude'
    var_28 = 'a.py,b.py,c.py'
    var_29 = [var_27, var_28, var_24]
    var_30 = module_0.make_config(var_29)



# Parsed testcases at query #90
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '--min-confidence'
    var_8 = '50'
    var_9 = '--verbose'
    var_10 = '--make-whitelist'
    var_11 = [var_3, var_7, var_8, var_9, var_10]
    var_12 = module_0.make_config(var_11)
    var_13 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_14 = []
    var_15 = 'cli_path'
    var_16 = '20'
    var_17 = [var_15, var_7, var_16]
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path1"]\n    '
    var_19 = []
    var_20 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path1"]\n    '
    var_21 = []
    var_22 = '\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_23 = []
    var_24 = '--version'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #91
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = b'\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    min_confidence = 70\n    verbose = true\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_5, var_11]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'\n    [tool.vulture]\n    unknown_key = "value"\n    paths = ["test.py"]\n    '
    var_16 = []
    var_17 = module_0.make_config(var_16, var_14)
    var_18 = b'\n    [tool.vulture]\n    paths = ["test.py"]\n    min_confidence = "high"\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19, var_14)
    var_21 = '--min-confidence'
    var_22 = 'not_a_number'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--config'
    var_26 = 'nonexistent.toml'
    var_27 = 'test.py'
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--make-whitelist'
    var_31 = '--sort-by-size'
    var_32 = [var_30, var_31, var_27]
    var_33 = module_0.make_config(var_32)
    var_34 = '--exclude'
    var_35 = '*.pyc,test_*.py'
    var_36 = '--ignore-decorators'
    var_37 = '@app.route,@require_*'
    var_38 = '--ignore-names'
    var_39 = 'visit_*,do_*'
    var_40 = 'main.py'
    var_41 = [var_34, var_35, var_36, var_37, var_38, var_39, var_40]
    var_42 = module_0.make_config(var_41)



# Parsed testcases at query #92
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_5, var_11, var_4]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = '--config'
    var_16 = 'test_path'
    var_17 = module_0.make_config(var_3)
    var_18 = b'[tool.vulture]\nunknown_key = true\n'
    var_19 = 'test'
    var_20 = [var_19]
    var_21 = b"[tool.vulture]\nmin_confidence = 'invalid'\n"
    var_22 = 'test'
    var_23 = [var_22]



# Parsed testcases at query #93
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--verbose'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'src'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = b'[tool.vulture]\nmin_confidence = 10\n'
    var_11 = '75'
    var_12 = [var_3, var_11]
    var_13 = []
    var_14 = module_0.make_config(var_13)
    var_15 = b'[tool.vulture]\ninvalid_key = 1\n'
    var_16 = []
    var_17 = b"[tool.vulture]\nmin_confidence = 'invalid'\n"
    var_18 = []
    var_19 = '[tool.vulture]\nmin_confidence = 42\n'
    var_20 = '--config'
    var_21 = 'test.py'
    var_22 = module_0.make_config(var_3)
    var_23 = '--version'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--help'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #94
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1.py'
    var_4 = 'path2.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = '--verbose'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 'cli_path.py'
    var_13 = '90'
    var_14 = [var_12, var_5, var_13]
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = '--invalid-option'
    var_19 = [var_18]
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)
    var_22 = '--min-confidence'
    var_23 = 'not_an_int'
    var_24 = [var_22, var_23]
    var_25 = None
    var_26 = module_0.make_config(var_24, var_25)
    var_27 = 'path.py'
    var_28 = '--config'
    var_29 = 'nonexistent.toml'
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.make_config(var_30, var_23)
    var_32 = '--make-whitelist'
    var_33 = [var_27, var_32]
    var_34 = module_0.make_config(var_33, var_23)
    var_35 = '--sort-by-size'
    var_36 = [var_27, var_35]
    var_37 = module_0.make_config(var_36, var_23)
    var_38 = '--exclude'
    var_39 = 'file1.py,file2.py'
    var_40 = [var_27, var_38, var_39]
    var_41 = module_0.make_config(var_40, var_23)
    var_42 = '--ignore-decorators'
    var_43 = '@app.route,@require_*'
    var_44 = [var_27, var_42, var_43]
    var_45 = module_0.make_config(var_44, var_23)
    var_46 = '--ignore-names'
    var_47 = 'visit_*,do_*'
    var_48 = [var_27, var_46, var_47]
    var_49 = module_0.make_config(var_48, var_23)



# Parsed testcases at query #95
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--verbose'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path"]\n    '
    var_12 = 'cli_path'
    var_13 = '20'
    var_14 = [var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    min_confidence = 10\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_20 = 'path'
    var_21 = [var_20]
    var_22 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_23 = 'path'
    var_24 = [var_23]



# Parsed testcases at query #96
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = 'path1.py'
    var_4 = 'path2.py'
    var_5 = '--min-confidence'
    var_6 = '50'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = []
    var_11 = '80'
    var_12 = [var_5, var_11]
    var_13 = b"[tool.vulture]\npaths = ['path1']"
    var_14 = []
    var_15 = []
    var_16 = b''
    var_17 = module_0.make_config(var_15, var_3)



# Parsed testcases at query #97
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8, var_1)
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = '20'
    var_13 = [var_3, var_12, var_5]
    var_14 = '--exclude'
    var_15 = 'file1.py,file2.py'
    var_16 = '--ignore-decorators'
    var_17 = 'deco1,deco2'
    var_18 = [var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--make-whitelist'
    var_21 = '--sort-by-size'
    var_22 = 'path'
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.make_config(var_23, var_1)
    var_25 = '--config'
    var_26 = 'nonexistent.toml'
    var_27 = [var_25, var_26, var_22]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_30 = []
    var_31 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_32 = []
    var_33 = []
    var_34 = None
    var_35 = module_0.make_config(var_33, var_34)



# Parsed testcases at query #98
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1.py'
    var_3 = 'path2.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_7 = 'test_config.toml'
    var_8 = []
    var_9 = '\n    [tool.vulture]\n    min_confidence = 10\n    paths = ["toml_path.py"]\n    '
    var_10 = 'override_config.toml'
    var_11 = 'cli_path.py'
    var_12 = '--min-confidence'
    var_13 = '20'
    var_14 = [var_11, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["path.py"]\n    '
    var_16 = 'verbose_config.toml'
    var_17 = []
    var_18 = 'pyproject.toml'
    var_19 = '\n    [tool.vulture]\n    paths = ["auto_detected.py"]\n    '
    var_20 = '--config'
    var_21 = 'invalid_key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = module_0._check_input_config(var_23)
    var_25 = 'min_confidence'
    var_26 = 'not_an_int'
    var_27 = {var_25: var_26}
    var_28 = module_0._check_input_config(var_27)



# Parsed testcases at query #99
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = '--verbose'
    var_5 = 'test.py'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_9 = []
    var_10 = '20'
    var_11 = [var_2, var_10, var_4]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = []
    var_15 = ''
    var_16 = module_0.make_config(var_14, var_3)
    var_17 = 'path1'
    var_18 = 'path2'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '\n        [tool.vulture]\n        min_confidence = 30\n        paths = ["test_path"]\n        '
    var_22 = '--config'
    var_23 = '--version'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--help'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #100
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '-v'
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = 'path1.py'
    var_6 = 'path2.py'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["test_*.py"]\n    min_confidence = 80\n    verbose = true\n    paths = ["src", "tests"]\n    '
    var_10 = []
    var_11 = '30'
    var_12 = [var_3, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = true\n    paths = ["test.py"]\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["test.py"]\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #101
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = '--min-confidence'
    var_3 = '50'
    var_4 = 'path1'
    var_5 = 'path2'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_9 = []
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1"]\n    '
    var_11 = '80'
    var_12 = 'cli_path'
    var_13 = [var_2, var_11, var_12]
    var_14 = '--make-whitelist'
    var_15 = '--sort-by-size'
    var_16 = '--verbose'
    var_17 = 'path'
    var_18 = [var_14, var_15, var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--exclude'
    var_21 = 'file1.py,file2.py'
    var_22 = [var_20, var_21, var_17]
    var_23 = module_0.make_config(var_22)
    var_24 = '--verbose'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '\n    [tool.vulture]\n    invalid_key = true\n    '
    var_28 = []
    var_29 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_30 = []



# Parsed testcases at query #102
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    exclude = ["excluded_file.py"]\n    verbose = true\n    '
    var_10 = []
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path"]\n    '
    var_12 = 'cli_path'
    var_13 = '80'
    var_14 = [var_12, var_4, var_13]
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = '\n    [tool.vulture]\n    invalid_key = "value"\n    paths = ["path"]\n    '
    var_18 = []
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    paths = ["path"]\n    '
    var_20 = []
    var_21 = '--config'
    var_22 = 'nonexistent.toml'
    var_23 = 'path'
    var_24 = [var_21, var_22, var_23]
    var_25 = module_0.make_config(var_24)



# Parsed testcases at query #103
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = []
    var_4 = None
    var_5 = module_0.make_config(var_3, var_4)
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_6, var_7]
    var_9 = module_0.make_config(var_8, var_4)
    var_10 = '--min-confidence'
    var_11 = '50'
    var_12 = '--exclude'
    var_13 = 'file1.py,file2.py'
    var_14 = '--ignore-decorators'
    var_15 = 'deco1,@deco2'
    var_16 = '--ignore-names'
    var_17 = 'name1,name2'
    var_18 = '--make-whitelist'
    var_19 = '--sort-by-size'
    var_20 = '--verbose'
    var_21 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_6]
    var_22 = module_0.make_config(var_21, var_4)
    var_23 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_24 = []
    var_25 = '80'
    var_26 = 'extra_path'
    var_27 = [var_10, var_25, var_20, var_26]
    var_28 = '\n[tool.vulture]\ninvalid_key = true\n'
    var_29 = 'path'
    var_30 = [var_29]
    var_31 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_32 = 'path'
    var_33 = [var_32]
    var_34 = '--config'
    var_35 = 'extra_path'
    var_36 = module_0.make_config(var_6)
    var_37 = '--config'
    var_38 = 'nonexistent.toml'
    var_39 = 'path'
    var_40 = [var_37, var_38, var_39]
    var_41 = module_0.make_config(var_40)
    var_42 = [var_20]



# Parsed testcases at query #104
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.make_config(var_0)
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_10 = 'utf-8'
    var_11 = '80'
    var_12 = [var_4, var_11, var_6]
    var_13 = []
    var_14 = b'[tool.vulture]\n'
    var_15 = module_0.make_config(var_13, var_3)
    var_16 = []
    var_17 = b'[tool.vulture]\ninvalid_key = 1\n'
    var_18 = module_0.make_config(var_16, var_3)
    var_19 = []
    var_20 = b"[tool.vulture]\nmin_confidence = 'not_int'\n"
    var_21 = module_0.make_config(var_19, var_3)



