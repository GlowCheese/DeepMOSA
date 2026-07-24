####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_18 = []
    var_19 = 'sort_by_size'
    var_20 = 30
    var_21 = 'toml_path1'
    var_22 = 'toml_path2'
    var_23 = [var_21, var_22]
    var_24 = {var_10: var_20, var_19: var_14, var_12: var_23}
    var_25 = '70'
    var_26 = 'cli_path'
    var_27 = [var_3, var_25, var_26]
    var_28 = 70
    var_29 = [var_26]
    var_30 = {var_10: var_28, var_19: var_14, var_12: var_29}
    var_31 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_32 = []
    var_33 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_34 = []
    var_35 = []
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = []
    var_10 = 0
    var_11 = '70'
    var_12 = [var_3, var_11]
    var_13 = 'path1'
    var_14 = 'path2'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15, var_1)
    var_17 = '--exclude'
    var_18 = 'test_*,venv'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19, var_1)
    var_21 = '--ignore-decorators'
    var_22 = 'deco1,deco2'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23, var_1)
    var_25 = '--ignore-names'
    var_26 = 'name1,name2'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = '--make-whitelist'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30, var_1)
    var_32 = '--sort-by-size'
    var_33 = [var_32]
    var_34 = module_0.make_config(var_33, var_1)
    var_35 = '--config'
    var_36 = 'custom.toml'
    var_37 = [var_35, var_36]
    var_38 = module_0.make_config(var_37, var_1)
    var_39 = '--version'
    var_40 = [var_39]
    var_41 = None
    var_42 = module_0.make_config(var_40, var_41)
    var_43 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_44 = []
    var_45 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_46 = []
    var_47 = []
    var_48 = None
    var_49 = module_0.make_config(var_47, var_48)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = 0
    var_11 = [var_3, var_4]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)
    var_17 = 'path1'
    var_18 = 'path2'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19, var_15)



# Parsed testcases at query #4
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
    var_10 = '\n        [tool.vulture]\n        min_confidence = 30\n        verbose = true\n        paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--version'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #5
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = '--exclude'
    var_20 = '*.py'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #6
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = '--exclude'
    var_20 = 'pattern'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #7
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = 0
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'test_*,docs'
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
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_42 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_43 = []
    var_44 = module_0.make_config(var_43)



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_9 = []
    var_10 = 0
    var_11 = '60'
    var_12 = [var_3, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)



# Parsed testcases at query #9
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    '
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = {var_10: var_20, var_19: var_22}
    var_24 = '\n    [tool.vulture]\n    min_confidence = 30\n    '
    var_25 = [var_3, var_4]
    var_26 = {var_10: var_13}
    var_27 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_28 = []
    var_29 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_30 = []
    var_31 = []
    var_32 = module_0.make_config(var_31)



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = 0
    var_11 = '70'
    var_12 = [var_3, var_11]
    var_13 = 'path1'
    var_14 = 'path2'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15, var_1)
    var_17 = '--exclude'
    var_18 = 'test_*,venv'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19, var_1)
    var_21 = '--ignore-decorators'
    var_22 = '@app.route,@require_*'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23, var_1)
    var_25 = '--ignore-names'
    var_26 = 'visit_*,do_*'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27, var_1)
    var_29 = '--make-whitelist'
    var_30 = [var_29]
    var_31 = module_0.make_config(var_30, var_1)
    var_32 = '--sort-by-size'
    var_33 = [var_32]
    var_34 = module_0.make_config(var_33, var_1)
    var_35 = '--config'
    var_36 = 'custom.toml'
    var_37 = [var_35, var_36]
    var_38 = module_0.make_config(var_37, var_1)
    var_39 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_40 = []
    var_41 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_42 = []
    var_43 = []
    var_44 = None
    var_45 = module_0.make_config(var_43, var_44)



# Parsed testcases at query #11
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = 0
    var_13 = 'cli_path'
    var_14 = [var_3, var_4, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--config'
    var_22 = 'nonexistent.toml'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["test.py"]\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = '[tool.vulture]\ninvalid_key = 123'
    var_13 = []
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_9 = []
    var_10 = '60'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*.py,venv'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = '@app.route,@require_*'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'visit_*,do_*'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_39 = []
    var_40 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_41 = []
    var_42 = []
    var_43 = None
    var_44 = module_0.make_config(var_42, var_43)



# Parsed testcases at query #14
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["path3", "path4"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'path5'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '--version'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = '--help'
    var_25 = [var_24]
    var_26 = module_0.make_config(var_25)



# Parsed testcases at query #15
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #16
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
    var_10 = '\n    [tool.vulture]\n    exclude = ["file*.py", "dir/"]\n    ignore_decorators = ["deco1", "deco2"]\n    ignore_names = ["name1", "name2"]\n    make_whitelist = true\n    min_confidence = 10\n    sort_by_size = true\n    verbose = true\n    paths = ["path1", "path2"]\n    '
    var_11 = []
    var_12 = 'path3'
    var_13 = [var_4, var_5, var_12]
    var_14 = '[tool.vulture]\ninvalid_key = 123'
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #17
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = '70'
    var_12 = '--exclude'
    var_13 = 'venv'
    var_14 = [var_3, var_11, var_12, var_13]
    var_15 = '[tool.vulture]\nunknown_key = 123'
    var_16 = "[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '[tool.vulture]\nverbose = true'



# Parsed testcases at query #18
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
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_9 = 0
    var_10 = '70'
    var_11 = [var_1, var_10]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_14 = module_0.make_config()



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 75\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = []
    var_12 = '30'
    var_13 = 'cli_path'
    var_14 = [var_4, var_12, var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)



# Parsed testcases at query #20
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*,docs'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = 'deco1,deco2'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'name1,name2'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_39 = []
    var_40 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_41 = []
    var_42 = []
    var_43 = None
    var_44 = module_0.make_config(var_42, var_43)



# Parsed testcases at query #21
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["test.py"]\n'
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_13 = []
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)



# Parsed testcases at query #22
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '90'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #23
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = 0
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '\n    [tool.vulture]\n    verbose = true\n    paths = ["path1"]\n    '
    var_20 = 'Reading configuration from <_io.BytesIO object at ...>'



# Parsed testcases at query #24
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = None
    var_21 = module_0.make_config(var_19, var_20)



# Parsed testcases at query #25
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_7 = 0
    var_8 = [var_1, var_2]
    var_9 = '[tool.vulture]\nunknown_key = 10'
    var_10 = '--exclude'
    var_11 = 'test*'
    var_12 = [var_10, var_11]
    var_13 = module_0.make_config(var_12)



# Parsed testcases at query #26
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = b'\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '70'
    var_27 = [var_3, var_26]
    var_28 = 70
    var_29 = [var_21]
    var_30 = [var_23]
    var_31 = {var_10: var_28, var_19: var_29, var_12: var_30}
    var_32 = b'\n[tool.vulture]\nunknown_key = "value"\n'
    var_33 = []
    var_34 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_35 = []
    var_36 = []
    var_37 = module_0.make_config(var_36)



# Parsed testcases at query #27
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*,docs'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = 'deco1,deco2'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'name1,name2'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = None
    var_41 = module_0.make_config(var_39, var_40)
    var_42 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_43 = []
    var_44 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_45 = []
    var_46 = []
    var_47 = None
    var_48 = module_0.make_config(var_46, var_47)



# Parsed testcases at query #28
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '60'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = module_0.make_config(var_16, var_1)
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19, var_1)
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = [var_6]
    var_24 = module_0.make_config(var_23)



# Parsed testcases at query #29
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
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nmin_confidence = 75\npaths = ["toml_path1", "toml_path2"]\n'
    var_10 = []
    var_11 = '60'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '-v'
    var_15 = [var_14]
    var_16 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_17 = []
    var_18 = module_0.make_config(var_17, var_3)
    var_19 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_20 = []
    var_21 = module_0.make_config(var_20, var_3)
    var_22 = []
    var_23 = '[tool.vulture]'
    var_24 = module_0.make_config(var_22, var_21)



# Parsed testcases at query #30
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\npaths = ["toml_path1"]\n'
    var_18 = []
    var_19 = 'sort_by_size'
    var_20 = 30
    var_21 = 'toml_path1'
    var_22 = [var_21]
    var_23 = {var_10: var_20, var_19: var_14, var_12: var_22}
    var_24 = '70'
    var_25 = '--sort-by-size'
    var_26 = [var_3, var_24, var_25]
    var_27 = 70
    var_28 = [var_21]
    var_29 = {var_10: var_27, var_19: var_14, var_12: var_28}
    var_30 = b'[tool.vulture]\ninvalid_key = 10'
    var_31 = []
    var_32 = b'[tool.vulture]\nmin_confidence = "not_an_int"'
    var_33 = []
    var_34 = []
    var_35 = None
    var_36 = module_0.make_config(var_34, var_35)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_9 = []
    var_10 = '100'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*.py,venv'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = '@app.route,@require_*'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'visit_*,do_*'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = [var_5]
    var_39 = module_0.make_config(var_38, var_1)
    var_40 = '--version'
    var_41 = [var_40]
    var_42 = None
    var_43 = module_0.make_config(var_41, var_42)
    var_44 = '--help'
    var_45 = [var_44]
    var_46 = None
    var_47 = module_0.make_config(var_45, var_46)
    var_48 = '--invalid-key'
    var_49 = 'value'
    var_50 = [var_48, var_49]
    var_51 = None
    var_52 = module_0.make_config(var_50, var_51)
    var_53 = '--min-confidence'
    var_54 = 'not-an-integer'
    var_55 = [var_53, var_54]
    var_56 = None
    var_57 = module_0.make_config(var_55, var_56)
    var_58 = []
    var_59 = None
    var_60 = module_0.make_config(var_58, var_59)



# Parsed testcases at query #32
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '70'
    var_27 = 'cli_path'
    var_28 = [var_3, var_26, var_27]
    var_29 = 70
    var_30 = [var_21]
    var_31 = [var_27]
    var_32 = {var_10: var_29, var_19: var_30, var_12: var_31}
    var_33 = '[tool.vulture]\ninvalid_key = 123'
    var_34 = []
    var_35 = []
    var_36 = None
    var_37 = module_0.make_config(var_35, var_36)



# Parsed testcases at query #33
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #34
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #35
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
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 30\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_10 = []
    var_11 = '70'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '-v'
    var_15 = [var_14]
    var_16 = '--invalid-key'
    var_17 = 'value'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18)
    var_20 = '--min-confidence'
    var_21 = 'not_an_int'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)
    var_24 = []
    var_25 = module_0.make_config(var_24)
    var_26 = [var_5]
    var_27 = module_0.make_config(var_26)
    var_28 = '\n    [tool.vulture]\n    paths = ["toml_path"]\n    '
    var_29 = '--exclude'
    var_30 = 'pattern1,pattern2'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--ignore-decorators'
    var_34 = 'deco1,deco2'
    var_35 = [var_33, var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = '--ignore-names'
    var_38 = 'name1,name2'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '--make-whitelist'
    var_42 = [var_41]
    var_43 = module_0.make_config(var_42)
    var_44 = '--sort-by-size'
    var_45 = [var_44]
    var_46 = module_0.make_config(var_45)
    var_47 = '--config'
    var_48 = 'custom.toml'
    var_49 = [var_47, var_48]
    var_50 = module_0.make_config(var_49)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '[tool.vulture]\ninvalid_key = 123'
    var_16 = []
    var_17 = '--min-confidence'
    var_18 = 'not_an_int'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = []
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #2
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 75\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '60'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #3
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_7 = '25'
    var_8 = [var_1, var_7]
    var_9 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_10 = '--exclude'
    var_11 = 'test_*.py'
    var_12 = [var_10, var_11]
    var_13 = module_0.make_config(var_12)
    var_14 = 'path1'
    var_15 = 'path2'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--exclude'
    var_19 = 'test_*.py,venv'
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
    var_31 = [var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--sort-by-size'
    var_34 = [var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = '--config'
    var_37 = 'custom.toml'
    var_38 = [var_36, var_37]
    var_39 = module_0.make_config(var_38)



# Parsed testcases at query #4
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = '70'
    var_12 = [var_3, var_11]
    var_13 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_14 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_15 = []
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #5
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    paths = ["src/"]\n    '
    var_11 = []
    var_12 = '100'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #6
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = [var_3, var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #7
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
    var_7 = 'min_confidence'
    var_8 = 'verbose'
    var_9 = 50
    var_10 = True
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_13 = 75
    var_14 = {var_7: var_13, var_8: var_10}
    var_15 = '60'
    var_16 = [var_2, var_15]
    var_17 = 60
    var_18 = {var_7: var_17, var_8: var_10}
    var_19 = 'path1'
    var_20 = 'path2'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--exclude'
    var_24 = 'test_*,venv'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = 'test_*'
    var_28 = 'venv'
    var_29 = '--ignore-decorators'
    var_30 = 'deco1,deco2'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = 'deco1'
    var_34 = 'deco2'
    var_35 = '--ignore-names'
    var_36 = 'name1,name2'
    var_37 = [var_35, var_36]
    var_38 = module_0.make_config(var_37)
    var_39 = 'name1'
    var_40 = 'name2'
    var_41 = '--make-whitelist'
    var_42 = [var_41]
    var_43 = module_0.make_config(var_42)
    var_44 = '--sort-by-size'
    var_45 = [var_44]
    var_46 = module_0.make_config(var_45)
    var_47 = '--config'
    var_48 = 'custom.toml'
    var_49 = [var_47, var_48]
    var_50 = module_0.make_config(var_49)
    var_51 = '--version'
    var_52 = [var_51]
    var_53 = module_0.make_config(var_52)
    var_54 = '\n    [tool.vulture]\n    invalid_key = 123\n    '
    var_55 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_56 = []
    var_57 = module_0.make_config(var_56)
    var_58 = module_0._check_output_config(var_57)



# Parsed testcases at query #8
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 75\nverbose = true\n'
    var_7 = [var_1, var_2]
    var_8 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_9 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_10 = []
    var_11 = module_0.make_config(var_10)
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'



# Parsed testcases at query #9
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_9 = []
    var_10 = '30'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*,docs'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = '@app.route,@require_*'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'visit_*,do_*'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = None
    var_41 = module_0.make_config(var_39, var_40)
    var_42 = '--help'
    var_43 = [var_42]
    var_44 = None
    var_45 = module_0.make_config(var_43, var_44)
    var_46 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_47 = []
    var_48 = '\n    [tool.vulture]\n    min_confidence = "not_an_integer"\n    '
    var_49 = []
    var_50 = []
    var_51 = None
    var_52 = module_0.make_config(var_50, var_51)



# Parsed testcases at query #10
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
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nmin_confidence = 30\npaths = ["toml_path1", "toml_path2"]\n'
    var_10 = []
    var_11 = 'cli_path'
    var_12 = [var_3, var_4, var_11]
    var_13 = '[tool.vulture]\ninvalid_key = value'
    var_14 = []
    var_15 = []
    var_16 = '[tool.vulture]'
    var_17 = module_0.make_config(var_15, var_3)



# Parsed testcases at query #11
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
    var_8 = module_0.make_config(var_7)
    var_9 = '\n[tool.vulture]\nmin_confidence = 75\npaths = ["toml_path1", "toml_path2"]\n'
    var_10 = []
    var_11 = '100'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["test.py"]\n'
    var_9 = 0
    var_10 = '60'
    var_11 = [var_3, var_10]
    var_12 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_13 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_14 = []
    var_15 = module_0.make_config(var_14)
    var_16 = 'test.py'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n[tool.vulture]\npaths = ["test.py"]\n'



# Parsed testcases at query #13
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = '--config'
    var_1 = 'nonexistent.toml'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--verbose'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    '
    var_10 = 'test_config.toml'
    var_11 = 'rb'
    var_12 = open(var_10, var_11)
    var_13 = module_0.make_config(tomlfile=var_12)
    var_14 = '30'
    var_15 = '--exclude'
    var_16 = '*.py'
    var_17 = [var_4, var_14, var_15, var_16]
    var_18 = open(var_10, var_11)
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_21 = 'test_config.toml'
    var_22 = 'rb'
    var_23 = open(var_21, var_22)
    var_24 = module_0.make_config(tomlfile=var_23)
    var_25 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_26 = 'test_config.toml'
    var_27 = 'rb'
    var_28 = open(var_26, var_27)
    var_29 = module_0.make_config(tomlfile=var_28)
    var_30 = '--min-confidence'
    var_31 = '50'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = 'test_config.toml'



# Parsed testcases at query #14
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = 'min_confidence'
    var_9 = 'verbose'
    var_10 = 50
    var_11 = True
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_14 = []
    var_15 = 30
    var_16 = {var_8: var_15, var_9: var_11}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_18 = [var_3, var_4]
    var_19 = {var_8: var_10, var_9: var_11}
    var_20 = '\n[tool.vulture]\ninvalid_key = 10\n'
    var_21 = []
    var_22 = []
    var_23 = None
    var_24 = module_0.make_config(var_22, var_23)
    var_25 = 'path1'
    var_26 = 'path2'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27, var_23)
    var_29 = 'paths'
    var_30 = [var_25, var_26]
    var_31 = {var_29: var_30}
    var_32 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'
    var_33 = []
    var_34 = [var_25, var_26]
    var_35 = {var_29: var_34}



# Parsed testcases at query #15
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["path3", "path4"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '60'
    var_14 = 'path5'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #16
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = false\n    paths = ["toml_path1"]\n    '
    var_13 = 'cli_path1'
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #17
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '70'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = []
    var_19 = None
    var_20 = module_0.make_config(var_18, var_19)



# Parsed testcases at query #18
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #19
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 75\nverbose = true\n'
    var_7 = '25'
    var_8 = [var_1, var_7]
    var_9 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_10 = module_0.make_config()



# Parsed testcases at query #20
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
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["deco1"]\nmin_confidence = 30\n'
    var_11 = []
    var_12 = '70'
    var_13 = [var_4, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = '--exclude'
    var_19 = 'test_*.py'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #21
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    '
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = {var_10: var_20, var_19: var_22}
    var_24 = '70'
    var_25 = [var_3, var_24]
    var_26 = 70
    var_27 = [var_21]
    var_28 = {var_10: var_26, var_19: var_27}
    var_29 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_30 = []
    var_31 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_32 = []
    var_33 = []
    var_34 = module_0.make_config(var_33)
    var_35 = [var_6]
    var_36 = module_0.make_config(var_35)



# Parsed testcases at query #22
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = [var_3, var_4, var_5, var_12]
    var_14 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = '--min-confidence'
    var_19 = '50'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #23
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = 'path1'
    var_1 = 'path2'
    var_2 = [var_0, var_1]
    var_3 = module_0.make_config(var_2)
    var_4 = '--min-confidence'
    var_5 = '50'
    var_6 = '--make-whitelist'
    var_7 = '--sort-by-size'
    var_8 = '--verbose'
    var_9 = '--exclude'
    var_10 = 'test_*,example.py'
    var_11 = '--ignore-decorators'
    var_12 = 'deco1,deco2'
    var_13 = '--ignore-names'
    var_14 = 'name1,name2'
    var_15 = [var_0, var_1, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n[tool.vulture]\nexclude = ["test_*", "example.py"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 50\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_18 = 'path3'
    var_19 = '75'
    var_20 = [var_18, var_4, var_19]
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_24 = module_0.make_config(tomlfile=var_21)
    var_25 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_26 = module_0.make_config(tomlfile=var_21)



# Parsed testcases at query #24
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '--exclude'
    var_17 = 'test_*,docs'
    var_18 = [var_16, var_17]
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = '--ignore-decorators'
    var_21 = 'deco1,deco2'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = '--ignore-names'
    var_25 = 'name1,name2'
    var_26 = [var_24, var_25]
    var_27 = module_0.make_config(var_26, var_1)
    var_28 = '--make-whitelist'
    var_29 = [var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32, var_1)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36, var_1)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = None
    var_41 = module_0.make_config(var_39, var_40)
    var_42 = '\n[tool.vulture]\ninvalid_key = 123\n'
    var_43 = []
    var_44 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_45 = []
    var_46 = []
    var_47 = None
    var_48 = module_0.make_config(var_46, var_47)



# Parsed testcases at query #25
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["path3", "path4"]\n    '
    var_11 = []
    var_12 = 'path5'
    var_13 = [var_3, var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #26
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
    var_7 = 'min_confidence'
    var_8 = 'verbose'
    var_9 = 50
    var_10 = True
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["src/"]\n'
    var_13 = 'paths'
    var_14 = 30
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = {var_7: var_14, var_8: var_10, var_13: var_16}
    var_18 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_19 = [var_2, var_3]
    var_20 = {var_7: var_9, var_8: var_10}
    var_21 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_22 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_23 = []
    var_24 = module_0.make_config(var_23)
    var_25 = [var_15]
    var_26 = module_0.make_config(var_25)
    var_27 = [var_15]
    var_28 = {var_13: var_27}



# Parsed testcases at query #27
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
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@decorator"]\nmin_confidence = 30\npaths = ["path3"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'path4'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = '--exclude'
    var_20 = 'test_*.py'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #28
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '60'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #29
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = 0
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #30
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0
import tomli._parser as module_1

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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = module_1.loads(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12, var_11)
    var_14 = '60'
    var_15 = 'cli_path'
    var_16 = [var_3, var_14, var_5, var_15]
    var_17 = module_0.make_config(var_16, var_11)
    var_18 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_19 = module_1.loads(var_18)
    var_20 = []
    var_21 = module_0.make_config(var_20, var_19)
    var_22 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_23 = module_1.loads(var_22)
    var_24 = []
    var_25 = module_0.make_config(var_24, var_23)
    var_26 = []
    var_27 = None
    var_28 = module_0.make_config(var_26, var_27)



# Parsed testcases at query #32
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = [var_3, var_4]
    var_11 = '\n    [tool.vulture]\n    invalid_key = 10\n    '
    var_12 = []
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_14 = []
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)



# Parsed testcases at query #33
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
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["deco1"]\nmin_confidence = 30\n'
    var_11 = []
    var_12 = '70'
    var_13 = [var_4, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = '--config'
    var_21 = 'nonexistent.toml'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22)



# Parsed testcases at query #34
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #35
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["path3", "path4"]\n    '
    var_11 = []
    var_12 = '60'
    var_13 = 'path5'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #36
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = []
    var_12 = 0
    var_13 = '70'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_17 = []
    var_18 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = '\n[tool.vulture]\npaths = ["src"]\n'
    var_23 = [var_5]



# Parsed testcases at query #37
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n        [tool.vulture]\n        min_confidence = 30\n        verbose = true\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'test_*,*.pyc'
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
    var_31 = '--sort-by-size'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--config'
    var_35 = 'custom.toml'
    var_36 = [var_34, var_35]
    var_37 = module_0.make_config(var_36)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = []
    var_42 = module_0.make_config(var_41)
    var_43 = '\n        [tool.vulture]\n        unknown_key = "value"\n    '
    var_44 = []
    var_45 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_46 = []



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nmin_confidence = 75\nverbose = true\n'
    var_9 = '60'
    var_10 = [var_3, var_9]
    var_11 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_12 = module_0.make_config(tomlfile=var_1)
    var_13 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_14 = module_0.make_config(tomlfile=var_1)
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = 'path1'
    var_18 = 'path2'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--exclude'
    var_22 = '*.py,test_*'
    var_23 = [var_21, var_22]
    var_24 = module_0.make_config(var_23)
    var_25 = '--ignore-decorators'
    var_26 = '@deco1,@deco2'
    var_27 = [var_25, var_26]
    var_28 = module_0.make_config(var_27)
    var_29 = '--ignore-names'
    var_30 = 'name1,name2'
    var_31 = [var_29, var_30]
    var_32 = module_0.make_config(var_31)
    var_33 = '--make-whitelist'
    var_34 = [var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = '--sort-by-size'
    var_37 = [var_36]
    var_38 = module_0.make_config(var_37)
    var_39 = '--config'
    var_40 = 'custom.toml'
    var_41 = [var_39, var_40]
    var_42 = module_0.make_config(var_41)



# Parsed testcases at query #39
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
    var_8 = 'min_confidence'
    var_9 = 'verbose'
    var_10 = 'paths'
    var_11 = 50
    var_12 = True
    var_13 = [var_4, var_5]
    var_14 = {var_8: var_11, var_9: var_12, var_10: var_13}
    var_15 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    '
    var_16 = 'exclude'
    var_17 = 75
    var_18 = 'test_*.py'
    var_19 = [var_18]
    var_20 = {var_8: var_17, var_16: var_19}
    var_21 = '25'
    var_22 = [var_1, var_21]
    var_23 = '\n    [tool.vulture]\n    min_confidence = 75\n    '
    var_24 = 25
    var_25 = {var_8: var_24}
    var_26 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_27 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_28 = '--min-confidence'
    var_29 = '50'
    var_30 = [var_28, var_29]
    var_31 = module_0.make_config(var_30)



# Parsed testcases at query #40
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = 0
    var_12 = '70'
    var_13 = [var_3, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\nverbose = true\npaths = ["src"]\n'
    var_19 = 'Reading configuration from <_io.StringIO object>'



# Parsed testcases at query #41
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
    var_10 = b'\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["path3", "path4"]\n'
    var_11 = []
    var_12 = 'path5'
    var_13 = [var_3, var_4, var_12]
    var_14 = b'\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = []
    var_16 = []
    var_17 = module_0.make_config(var_16)



# Parsed testcases at query #42
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #43
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '70'
    var_27 = 'cli_path'
    var_28 = [var_3, var_26, var_27]
    var_29 = 70
    var_30 = [var_21]
    var_31 = [var_27]
    var_32 = {var_10: var_29, var_19: var_30, var_12: var_31}
    var_33 = '[tool.vulture]\ninvalid_key = 123'
    var_34 = []
    var_35 = "[tool.vulture]\nmin_confidence = 'not_a_number'"
    var_36 = []
    var_37 = '--min-confidence'
    var_38 = '50'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)



# Parsed testcases at query #44
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["path3", "path4"]\n    '
    var_11 = []
    var_12 = '60'
    var_13 = 'path5'
    var_14 = [var_3, var_12, var_13]
    var_15 = '[tool.vulture]\nunknown_key = 10'
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)



# Parsed testcases at query #45
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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '70'
    var_27 = 'cli_path'
    var_28 = [var_3, var_26, var_27]
    var_29 = 70
    var_30 = [var_21]
    var_31 = [var_27]
    var_32 = {var_10: var_29, var_19: var_30, var_12: var_31}
    var_33 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_34 = []
    var_35 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_36 = []
    var_37 = '--exclude'
    var_38 = 'test_*.py'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)



