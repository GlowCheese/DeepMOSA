####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10, var_5]
    var_12 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_13 = []
    var_14 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_15 = []
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)



# Parsed testcases at query #2
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = b'\n[tool.vulture]\nmin_confidence = 75\nverbose = true\n'
    var_7 = '100'
    var_8 = [var_1, var_7]
    var_9 = b'\n[tool.vulture]\nunknown_key = "value"\n'
    var_10 = b'\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_11 = '--exclude'
    var_12 = 'test_*.py'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = 'path1'
    var_16 = 'path2'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--exclude'
    var_20 = 'test_*.py,venv'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)
    var_23 = '--ignore-decorators'
    var_24 = 'deco1,deco2'
    var_25 = [var_23, var_24]
    var_26 = module_0.make_config(var_25)
    var_27 = '--ignore-names'
    var_28 = 'name1,name2'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29)
    var_31 = '--make-whitelist'
    var_32 = [var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--sort-by-size'
    var_35 = [var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = b'\n[tool.vulture]\nmin_confidence = 25\n'
    var_38 = '--config'
    var_39 = 'custom.toml'
    var_40 = [var_38, var_39]



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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = '70'
    var_12 = '--exclude'
    var_13 = 'venv'
    var_14 = [var_3, var_11, var_12, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = module_0.make_config(var_17)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = '--min-confidence'
    var_12 = '100'
    var_13 = [var_11, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = 'test.toml'
    var_18 = 'invalid_test.toml'



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = 'min_confidence'
    var_9 = 'verbose'
    var_10 = 50
    var_11 = True
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_14 = []
    var_15 = 75
    var_16 = {var_8: var_15, var_9: var_11}
    var_17 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_18 = [var_3, var_4]
    var_19 = {var_8: var_10, var_9: var_11}
    var_20 = 'path1'
    var_21 = 'path2'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_1)
    var_24 = 'paths'
    var_25 = [var_20, var_21]
    var_26 = {var_24: var_25}
    var_27 = '--exclude'
    var_28 = 'test_*,*.pyc'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29, var_1)
    var_31 = 'exclude'
    var_32 = 'test_*'
    var_33 = '*.pyc'
    var_34 = [var_32, var_33]
    var_35 = {var_31: var_34}
    var_36 = '--ignore-decorators'
    var_37 = 'deco1,deco2'
    var_38 = [var_36, var_37]
    var_39 = module_0.make_config(var_38, var_1)
    var_40 = 'ignore_decorators'
    var_41 = 'deco1'
    var_42 = 'deco2'
    var_43 = [var_41, var_42]
    var_44 = {var_40: var_43}
    var_45 = '--ignore-names'
    var_46 = 'name1,name2'
    var_47 = [var_45, var_46]
    var_48 = module_0.make_config(var_47, var_1)
    var_49 = 'ignore_names'
    var_50 = 'name1'
    var_51 = 'name2'
    var_52 = [var_50, var_51]
    var_53 = {var_49: var_52}
    var_54 = '--make-whitelist'
    var_55 = [var_54]
    var_56 = module_0.make_config(var_55, var_1)
    var_57 = 'make_whitelist'
    var_58 = {var_57: var_11}
    var_59 = '--sort-by-size'
    var_60 = [var_59]
    var_61 = module_0.make_config(var_60, var_1)
    var_62 = 'sort_by_size'
    var_63 = {var_62: var_11}
    var_64 = '--config'
    var_65 = 'custom.toml'
    var_66 = [var_64, var_65]
    var_67 = module_0.make_config(var_66, var_1)
    var_68 = 'config'
    var_69 = {var_68: var_65}
    var_70 = '--version'
    var_71 = [var_70]
    var_72 = None
    var_73 = module_0.make_config(var_71, var_72)
    var_74 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_75 = []
    var_76 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_77 = []
    var_78 = []
    var_79 = None
    var_80 = module_0.make_config(var_78, var_79)



# Parsed testcases at query #6
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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = module_1.loads(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12, var_11)
    var_14 = '70'
    var_15 = 'cli_path1'
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
    var_27 = module_0.make_config(var_26)



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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nignore_names = ["test_*"]\n'
    var_11 = []
    var_12 = 0
    var_13 = '70'
    var_14 = [var_3, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '\n[tool.vulture]\npaths = ["path1"]\n'
    var_22 = '--verbose'
    var_23 = [var_22]
    var_24 = 'Reading configuration from <_io.StringIO object>'



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
    var_6 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    '
    var_7 = 0
    var_8 = '25'
    var_9 = [var_1, var_8]
    var_10 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_11 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_12 = '--exclude'
    var_13 = 'test_*.py'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)



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
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = 0
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = 'path1'
    var_13 = 'path2'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--exclude'
    var_17 = 'test_*.py,venv'
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
    var_38 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_39 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_40 = []
    var_41 = module_0.make_config(var_40)



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
    var_8 = 'min_confidence'
    var_9 = 'verbose'
    var_10 = 50
    var_11 = True
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_14 = []
    var_15 = 30
    var_16 = {var_8: var_15, var_9: var_11}
    var_17 = '70'
    var_18 = [var_3, var_17]
    var_19 = 70
    var_20 = {var_8: var_19, var_9: var_11}
    var_21 = '[tool.vulture]\nunknown_key = 123'
    var_22 = []
    var_23 = []
    var_24 = None
    var_25 = module_0.make_config(var_23, var_24)



# Parsed testcases at query #11
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
    var_7 = '\n[tool.vulture]\nmin_confidence = 30\npaths = ["toml_path1", "toml_path2"]\n'
    var_8 = [var_1, var_2]
    var_9 = b'invalid toml content'
    var_10 = b'[tool.vulture]\nunknown_key = 123'
    var_11 = b"[tool.vulture]\nmin_confidence = 'not_an_int'"
    var_12 = '--min-confidence'
    var_13 = '50'
    var_14 = [var_12, var_13]
    var_15 = module_0.make_config(var_14)
    var_16 = '--verbose'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '--config'
    var_20 = 'custom.toml'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #12
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 30\nignore_names = ["test_*"]\n'
    var_7 = '70'
    var_8 = [var_1, var_7]
    var_9 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_10 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_11 = module_0.make_config()
    var_12 = '--verbose'
    var_13 = [var_12]



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
    var_7 = module_0.make_config(var_6)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = 0
    var_10 = [var_3, var_4]
    var_11 = '\n[tool.vulture]\ninvalid_key = 10\n'
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = 'path1'
    var_15 = 'path2'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'



# Parsed testcases at query #14
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
    var_10 = 'cli_path1'
    var_11 = [var_3, var_4, var_10]
    var_12 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_13 = module_0.make_config(tomlfile=var_0)
    var_14 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_15 = module_0.make_config(tomlfile=var_0)
    var_16 = []
    var_17 = module_0.make_config(var_16)



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
    var_33 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_34 = []
    var_35 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_36 = []
    var_37 = '--exclude'
    var_38 = 'test.py'
    var_39 = [var_37, var_38]
    var_40 = module_0.make_config(var_39)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["path3"]\n    '
    var_11 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["path3"]\n    '
    var_12 = [var_3, var_4, var_6]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_15 = []
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #17
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
    var_7 = '70'
    var_8 = [var_1, var_7]
    var_9 = '[tool.vulture]\ninvalid_key = 123'
    var_10 = module_0.make_config()
    var_11 = 'path1'
    var_12 = 'path2'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)



# Parsed testcases at query #18
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
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@decorator"]\nmin_confidence = 30\n'
    var_11 = []
    var_12 = '--min-confidence'
    var_13 = '70'
    var_14 = [var_12, var_13]
    var_15 = 'invalid toml data'
    var_16 = []
    var_17 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_18 = []
    var_19 = '\n[tool.vulture]\nmin_confidence = "not an int"\n'
    var_20 = []
    var_21 = []
    var_22 = module_0.make_config(var_21)
    var_23 = '--version'
    var_24 = [var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--help'
    var_27 = [var_26]
    var_28 = module_0.make_config(var_27)



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
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
    var_10 = 'test_*,venv'
    var_11 = '--ignore-decorators'
    var_12 = 'deco1,deco2'
    var_13 = '--ignore-names'
    var_14 = 'name1,name2'
    var_15 = [var_0, var_1, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '\n[tool.vulture]\nexclude = ["file*.py", "dir/"]\nignore_decorators = ["deco1", "deco2"]\nignore_names = ["name1", "name2"]\nmake_whitelist = true\nmin_confidence = 10\nsort_by_size = true\nverbose = true\npaths = ["path1", "path2"]\n'
    var_18 = 'path3'
    var_19 = '30'
    var_20 = [var_18, var_4, var_19]
    var_21 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_22 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_23 = []
    var_24 = module_0.make_config(var_23)



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
    var_8 = 'min_confidence'
    var_9 = 'verbose'
    var_10 = 50
    var_11 = True
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '\n[tool.vulture]\nmin_confidence = 75\nverbose = true\n'
    var_14 = []
    var_15 = 75
    var_16 = {var_8: var_15, var_9: var_11}
    var_17 = 0
    var_18 = [var_3, var_4]
    var_19 = {var_8: var_10, var_9: var_11}
    var_20 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_21 = []
    var_22 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_23 = []
    var_24 = []
    var_25 = None
    var_26 = module_0.make_config(var_24, var_25)
    var_27 = 'path1'
    var_28 = 'path2'
    var_29 = [var_27, var_28]
    var_30 = module_0.make_config(var_29, var_25)
    var_31 = 'paths'
    var_32 = [var_27, var_28]
    var_33 = {var_31: var_32}
    var_34 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'
    var_35 = []
    var_36 = [var_27, var_28]
    var_37 = {var_31: var_36}



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
    var_12 = [var_3, var_4, var_5, var_6]
    var_13 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = '--min-confidence'
    var_18 = '50'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '60'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nignore_names = ["test_*"]\npaths = ["dir1", "dir2"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)



# Parsed testcases at query #25
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
    var_15 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_16 = 'exclude'
    var_17 = 30
    var_18 = 'test_*.py'
    var_19 = [var_18]
    var_20 = 'src'
    var_21 = [var_20]
    var_22 = {var_8: var_17, var_16: var_19, var_10: var_21}
    var_23 = '\n[tool.vulture]\nmin_confidence = 30\n'
    var_24 = '70'
    var_25 = [var_1, var_24]
    var_26 = 70
    var_27 = {var_8: var_26}
    var_28 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_29 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_30 = '--min-confidence'
    var_31 = '50'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = '70'
    var_12 = [var_3, var_11]
    var_13 = '[tool.vulture]\ninvalid_key = 123'
    var_14 = "[tool.vulture]\nmin_confidence = 'not_a_number'"
    var_15 = []
    var_16 = module_0.make_config(var_15)



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
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = 0
    var_11 = [var_3, var_4]
    var_12 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_13 = []
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\n'
    var_9 = []
    var_10 = '--min-confidence'
    var_11 = '70'
    var_12 = [var_10, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = []
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)
    var_20 = 'path1'
    var_21 = 'path2'
    var_22 = [var_20, var_21]
    var_23 = module_0.make_config(var_22, var_18)
    var_24 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'
    var_25 = []



# Parsed testcases at query #29
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
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\nignore_decorators = ["@decorator1"]\nignore_names = ["name1"]\nmake_whitelist = true\nsort_by_size = true\nverbose = true\npaths = ["path3", "path4"]\n'
    var_9 = '70'
    var_10 = [var_1, var_9, var_3]
    var_11 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_12 = module_0.make_config(tomlfile=var_2)
    var_13 = '\n[tool.vulture]\nmin_confidence = "not_an_integer"\n'
    var_14 = module_0.make_config(tomlfile=var_2)
    var_15 = module_0.make_config()



# Parsed testcases at query #30
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = module_0.make_config(var_0, var_1)
    var_3 = '--min-confidence'
    var_4 = '10'
    var_5 = 'path1'
    var_6 = 'path2'
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.make_config(var_7)
    var_9 = '\n        [tool.vulture]\n        min_confidence = 20\n        paths = ["toml_path1", "toml_path2"]\n    '
    var_10 = '30'
    var_11 = 'cli_path'
    var_12 = [var_3, var_10, var_11]
    var_13 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_14 = module_0.make_config(tomlfile=var_0)
    var_15 = '\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_16 = module_0.make_config(tomlfile=var_0)
    var_17 = []
    var_18 = '[tool.vulture]'
    var_19 = module_0.make_config(var_17, var_3)



# Parsed testcases at query #31
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n        [tool.vulture]\n        min_confidence = 30\n        verbose = true\n    '
    var_7 = [var_1, var_2]
    var_8 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_9 = '--invalid-arg'
    var_10 = [var_9]
    var_11 = module_0.make_config(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12)
    var_14 = 'path1'
    var_15 = 'path2'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '\n        [tool.vulture]\n        paths = ["path1", "path2"]\n    '



# Parsed testcases at query #32
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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_9 = []
    var_10 = [var_3, var_4]
    var_11 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_12 = []
    var_13 = []
    var_14 = None
    var_15 = module_0.make_config(var_13, var_14)



# Parsed testcases at query #34
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '10'
    var_3 = 'path1'
    var_4 = 'path2'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.make_config(var_5)
    var_7 = '\n[tool.vulture]\nmin_confidence = 20\npaths = ["path3", "path4"]\n'
    var_8 = '30'
    var_9 = 'path5'
    var_10 = [var_1, var_8, var_9]
    var_11 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_12 = module_0.make_config(tomlfile=var_1)
    var_13 = '--min-confidence'
    var_14 = 'not_a_number'
    var_15 = [var_13, var_14]
    var_16 = module_0.make_config(var_15)
    var_17 = '--min-confidence'
    var_18 = '10'
    var_19 = [var_17, var_18]
    var_20 = module_0.make_config(var_19)
    var_21 = '--verbose'
    var_22 = [var_21]
    var_23 = module_0.make_config(var_22)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = '70'
    var_12 = '--sort-by-size'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = module_0.make_config(var_16)



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
    var_10 = b'\n        [tool.vulture]\n        min_confidence = 75\n        verbose = true\n        paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = module_0.make_config(var_11, var_10)
    var_13 = [var_3, var_4, var_5, var_6]
    var_14 = module_0.make_config(var_13, var_10)
    var_15 = b'\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_16 = []
    var_17 = module_0.make_config(var_16, var_15)
    var_18 = b'\n        [tool.vulture]\n        min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = module_0.make_config(var_19, var_18)
    var_21 = '--min-confidence'
    var_22 = '50'
    var_23 = [var_21, var_22]
    var_24 = None
    var_25 = module_0.make_config(var_23, var_24)



# Parsed testcases at query #37
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
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\npaths = ["toml_path1", "toml_path2"]\n'
    var_9 = 'cli_path'
    var_10 = [var_2, var_3, var_9]
    var_11 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_12 = module_0.make_config(tomlfile=var_0)
    var_13 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_14 = module_0.make_config(tomlfile=var_0)
    var_15 = []
    var_16 = module_0.make_config(var_15)



# Parsed testcases at query #38
#--------------------------


import vulture.config as module_0
import tomli._parser as module_1

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_7 = module_1.loads(var_6)
    var_8 = module_0.make_config(tomlfile=var_7)
    var_9 = [var_1, var_2]
    var_10 = module_0.make_config(var_9, var_7)
    var_11 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_12 = module_1.loads(var_11)
    var_13 = module_0.make_config(tomlfile=var_12)
    var_14 = '--exclude'
    var_15 = 'test*.py'
    var_16 = [var_14, var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = 'path1'
    var_19 = 'path2'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20)
    var_22 = '--exclude'
    var_23 = 'test*.py,venv'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24)
    var_26 = '--ignore-decorators'
    var_27 = 'deco1,deco2'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28)
    var_30 = '--ignore-names'
    var_31 = 'name1,name2'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32)
    var_34 = '--make-whitelist'
    var_35 = [var_34]
    var_36 = module_0.make_config(var_35)
    var_37 = '--sort-by-size'
    var_38 = [var_37]
    var_39 = module_0.make_config(var_38)
    var_40 = '--config'
    var_41 = 'custom.toml'
    var_42 = [var_40, var_41]
    var_43 = module_0.make_config(var_42)



# Parsed testcases at query #39
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
    var_11 = '70'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '[tool.vulture]\ninvalid_key = 123'
    var_15 = []
    var_16 = "[tool.vulture]\nmin_confidence = 'not_a_number'"
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = "[tool.vulture]\npaths = ['test.py']"
    var_21 = '--verbose'
    var_22 = [var_21]



# Parsed testcases at query #40
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
    var_7 = 'test.toml'
    var_8 = 'rb'
    var_9 = open(var_7, var_8)
    var_10 = module_0.make_config(tomlfile=var_9)
    var_11 = [var_1, var_2]
    var_12 = open(var_7, var_8)
    var_13 = module_0.make_config(var_11, var_12)
    var_14 = '\n    [tool.vulture]\n    invalid_key = 123\n    '
    var_15 = 'test.toml'
    var_16 = 'rb'
    var_17 = open(var_15, var_16)
    var_18 = module_0.make_config(tomlfile=var_17)
    var_19 = '\n    [tool.vulture]\n    min_confidence = "not_a_number"\n    '
    var_20 = 'test.toml'
    var_21 = 'rb'
    var_22 = open(var_20, var_21)
    var_23 = module_0.make_config(tomlfile=var_22)
    var_24 = module_0.make_config()



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '70'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = '--sort-by-size'
    var_14 = 'cli_path1'
    var_15 = [var_3, var_12, var_13, var_14]
    var_16 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_17 = []
    var_18 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_19 = []
    var_20 = '\n[tool.vulture]\nmin_confidence = 50\n'
    var_21 = []



# Parsed testcases at query #43
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
    var_11 = 'cli_path1'
    var_12 = [var_3, var_4, var_11]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)
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
    var_3 = '--min-confidence'
    var_4 = '50'
    var_5 = '--verbose'
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 75\nsort_by_size = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '--min-confidence'
    var_13 = '30'
    var_14 = 'cli_path'
    var_15 = [var_12, var_13, var_14]
    var_16 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_17 = []
    var_18 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = 'test_pyproject.toml'
    var_23 = 'invalid_pyproject.toml'
    var_24 = 'wrong_type_pyproject.toml'



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = '70'
    var_12 = '--exclude'
    var_13 = 'venv'
    var_14 = [var_3, var_11, var_12, var_13]
    var_15 = 'test_pyproject.toml'
    var_16 = 'rb'
    var_17 = open(var_15, var_16)
    var_18 = module_0.make_config(var_14, var_17)
    var_19 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_20 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_21 = []
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #46
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
    var_10 = '60'
    var_11 = 'cli_path'
    var_12 = [var_3, var_10, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = module_0.make_config(tomlfile=var_0)
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = module_0.make_config(tomlfile=var_0)
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '-v'
    var_20 = [var_19, var_5]
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #47
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
    var_10 = '--min-confidence'
    var_11 = '70'
    var_12 = [var_10, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_15 = []
    var_16 = module_0.make_config(var_15)
    var_17 = 'test_pyproject.toml'
    var_18 = 'invalid_pyproject.toml'
    var_19 = 'wrong_type_pyproject.toml'



# Parsed testcases at query #48
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
    var_17 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_18 = []
    var_19 = 'exclude'
    var_20 = 30
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '70'
    var_27 = [var_3, var_26, var_5]
    var_28 = 70
    var_29 = [var_21]
    var_30 = [var_23]
    var_31 = {var_10: var_28, var_11: var_14, var_19: var_29, var_12: var_30}
    var_32 = '--invalid-arg'
    var_33 = 'value'
    var_34 = [var_32, var_33]
    var_35 = module_0.make_config(var_34)
    var_36 = "[tool.vulture]\ninvalid_key = 'value'"
    var_37 = []
    var_38 = []
    var_39 = module_0.make_config(var_38)
    var_40 = '--verbose'
    var_41 = [var_40]



# Parsed testcases at query #49
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
    var_13 = [var_3, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = []
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



# Parsed testcases at query #50
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
    var_10 = module_0.make_config(tomlfile=var_1)
    var_11 = '--exclude'
    var_12 = 'test_*.py'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '--version'
    var_16 = [var_15]
    var_17 = module_0.make_config(var_16)
    var_18 = '--help'
    var_19 = [var_18]
    var_20 = module_0.make_config(var_19)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = [var_3, var_4, var_5, var_6, var_7]
    var_13 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_14 = []
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = []
    var_18 = module_0.make_config(var_17)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = '--min-confidence'
    var_20 = '50'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = '70'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = module_0.make_config(var_16)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = '--exclude'
    var_14 = 'venv'
    var_15 = 'cli_path'
    var_16 = [var_3, var_12, var_13, var_14, var_15]
    var_17 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = '\n[tool.vulture]\npaths = ["src"]\nverbose = true\n'
    var_22 = []



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
    var_10 = '\n[tool.vulture]\nexclude = ["test_*.py"]\nignore_decorators = ["@decorator1"]\nmin_confidence = 30\n'
    var_11 = []
    var_12 = '70'
    var_13 = [var_3, var_12, var_5]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = []
    var_16 = module_0.make_config(var_15, var_1)
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = module_0.make_config(var_18, var_1)
    var_20 = []
    var_21 = '[tool.vulture]'
    var_22 = module_0.make_config(var_20, var_19)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    exclude = ["test_*.py"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = [var_3, var_4]
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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = []
    var_12 = 0
    var_13 = '70'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_17 = []
    var_18 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = [var_6]
    var_23 = module_0.make_config(var_22)



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



# Parsed testcases at query #10
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = 'min_confidence'
    var_7 = 'verbose'
    var_8 = 50
    var_9 = True
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = '\n    [tool.vulture]\n    min_confidence = 50\n    verbose = true\n    '
    var_12 = {var_6: var_8, var_7: var_9}
    var_13 = '70'
    var_14 = [var_1, var_13]
    var_15 = 70
    var_16 = {var_6: var_15, var_7: var_9}
    var_17 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = module_0.make_config()



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_9 = []
    var_10 = [var_3, var_4]
    var_11 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_12 = []
    var_13 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_14 = []
    var_15 = []
    var_16 = None
    var_17 = module_0.make_config(var_15, var_16)
    var_18 = 'path1'
    var_19 = 'path2'
    var_20 = [var_18, var_19]
    var_21 = module_0.make_config(var_20, var_16)
    var_22 = '--exclude'
    var_23 = 'test_*.py,venv'
    var_24 = [var_22, var_23]
    var_25 = module_0.make_config(var_24, var_16)
    var_26 = '--ignore-decorators'
    var_27 = '@app.route,@require_*'
    var_28 = [var_26, var_27]
    var_29 = module_0.make_config(var_28, var_16)
    var_30 = '--ignore-names'
    var_31 = 'visit_*,do_*'
    var_32 = [var_30, var_31]
    var_33 = module_0.make_config(var_32, var_16)
    var_34 = '--make-whitelist'
    var_35 = [var_34]
    var_36 = module_0.make_config(var_35, var_16)
    var_37 = '--sort-by-size'
    var_38 = [var_37]
    var_39 = module_0.make_config(var_38, var_16)
    var_40 = '--config'
    var_41 = 'custom.toml'
    var_42 = [var_40, var_41]
    var_43 = module_0.make_config(var_42, var_16)



# Parsed testcases at query #12
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
    var_11 = '70'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_15 = []
    var_16 = module_0.make_config(var_15, var_3)
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = module_0.make_config(var_18, var_3)
    var_20 = []
    var_21 = module_0.make_config(var_20)
    var_22 = '\n[tool.vulture]\npaths = ["some_path"]\n'
    var_23 = '--verbose'
    var_24 = [var_23]



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
    var_17 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_18 = []
    var_19 = 75
    var_20 = 'toml_path1'
    var_21 = 'toml_path2'
    var_22 = [var_20, var_21]
    var_23 = {var_10: var_19, var_11: var_14, var_12: var_22}
    var_24 = '60'
    var_25 = 'cli_path'
    var_26 = [var_3, var_24, var_25]
    var_27 = 60
    var_28 = [var_25]
    var_29 = {var_10: var_27, var_11: var_14, var_12: var_28}
    var_30 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_31 = []
    var_32 = module_0.make_config(var_31, var_3)
    var_33 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_34 = []
    var_35 = module_0.make_config(var_34, var_3)
    var_36 = []
    var_37 = module_0.make_config(var_36)
    var_38 = '--version'
    var_39 = [var_38]
    var_40 = module_0.make_config(var_39)
    var_41 = '--help'
    var_42 = [var_41]
    var_43 = module_0.make_config(var_42)



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
    var_10 = 'min_confidence'
    var_11 = 'verbose'
    var_12 = 'paths'
    var_13 = 50
    var_14 = True
    var_15 = [var_6, var_7]
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_18 = []
    var_19 = 30
    var_20 = 'toml_path1'
    var_21 = 'toml_path2'
    var_22 = [var_20, var_21]
    var_23 = {var_10: var_19, var_11: var_14, var_12: var_22}
    var_24 = '70'
    var_25 = 'cli_path'
    var_26 = [var_3, var_24, var_25]
    var_27 = 70
    var_28 = [var_25]
    var_29 = {var_10: var_27, var_11: var_14, var_12: var_28}
    var_30 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_31 = []
    var_32 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_33 = []
    var_34 = []
    var_35 = module_0.make_config(var_34)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["dir1"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = '--exclude'
    var_14 = 'venv'
    var_15 = 'cli_path'
    var_16 = [var_3, var_12, var_13, var_14, var_15]
    var_17 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_18 = []
    var_19 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_20 = []
    var_21 = []
    var_22 = module_0.make_config(var_21)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 75\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = []
    var_12 = 0
    var_13 = '30'
    var_14 = 'cli_path'
    var_15 = [var_3, var_13, var_14]
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



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
    var_16 = '\n    [tool.vulture]\n    unknown_key = "value"\n    '
    var_17 = []
    var_18 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_19 = []
    var_20 = []
    var_21 = module_0.make_config(var_20)



# Parsed testcases at query #18
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 75\nexclude = ["test_*.py"]\n'
    var_7 = '\n[tool.vulture]\nmin_confidence = 75\n'
    var_8 = [var_1, var_2]
    var_9 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_10 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_11 = '--exclude'
    var_12 = 'test_*.py'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = 'path1'
    var_16 = 'path2'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)



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
    var_6 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    '
    var_7 = [var_1, var_2]
    var_8 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_9 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_10 = module_0.make_config()
    var_11 = 'path1'
    var_12 = 'path2'
    var_13 = [var_11, var_12]
    var_14 = module_0.make_config(var_13)
    var_15 = '\n    [tool.vulture]\n    paths = ["path1", "path2"]\n    '



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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n    [tool.vulture]\n    min_confidence = 30\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = 'cli_path'
    var_13 = [var_3, var_4, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)



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
    var_12 = [var_3, var_4, var_5, var_6, var_7]
    var_13 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_14 = []
    var_15 = '--min-confidence'
    var_16 = '50'
    var_17 = [var_15, var_16]
    var_18 = module_0.make_config(var_17)



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["test.py"]\n'
    var_9 = []
    var_10 = 0
    var_11 = '70'
    var_12 = [var_3, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = []
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)



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
    var_8 = '\n        [tool.vulture]\n        min_confidence = 30\n        verbose = true\n    '
    var_9 = []
    var_10 = '70'
    var_11 = [var_3, var_10]
    var_12 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_13 = []
    var_14 = []
    var_15 = None
    var_16 = module_0.make_config(var_14, var_15)



# Parsed testcases at query #25
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
    var_10 = '60'
    var_11 = 'cli_path'
    var_12 = [var_3, var_10, var_11]
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = module_0.make_config(tomlfile=var_0)
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = module_0.make_config(tomlfile=var_0)
    var_17 = []
    var_18 = module_0.make_config(var_17)
    var_19 = '--verbose'
    var_20 = [var_19]



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
    var_17 = '\n    [tool.vulture]\n    min_confidence = 75\n    exclude = ["test_*.py"]\n    paths = ["src/"]\n    '
    var_18 = []
    var_19 = 'exclude'
    var_20 = 75
    var_21 = 'test_*.py'
    var_22 = [var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = {var_10: var_20, var_19: var_22, var_12: var_24}
    var_26 = '60'
    var_27 = [var_3, var_26, var_5]
    var_28 = 60
    var_29 = [var_21]
    var_30 = [var_23]
    var_31 = {var_10: var_28, var_11: var_14, var_19: var_29, var_12: var_30}
    var_32 = '[tool.vulture]\ninvalid_key = 123'
    var_33 = []
    var_34 = []
    var_35 = None
    var_36 = module_0.make_config(var_34, var_35)
    var_37 = '--version'
    var_38 = [var_37]
    var_39 = module_0.make_config(var_38)
    var_40 = '--help'
    var_41 = [var_40]
    var_42 = module_0.make_config(var_41)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_13]
    var_15 = '[tool.vulture]\nunknown_key = 123'
    var_16 = []
    var_17 = []
    var_18 = None
    var_19 = module_0.make_config(var_17, var_18)



# Parsed testcases at query #28
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
    var_13 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_14 = []
    var_15 = module_0.make_config(var_14, var_1)
    var_16 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_17 = []
    var_18 = module_0.make_config(var_17, var_1)
    var_19 = []
    var_20 = '[tool.vulture]'
    var_21 = module_0.make_config(var_19, var_18)



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["dir1", "dir2"]\n'
    var_11 = '70'
    var_12 = 'path3'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_15 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_16 = []
    var_17 = module_0.make_config(var_16)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '60'
    var_13 = 'cli_path'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["src/"]\n'
    var_11 = module_1.loads(var_10)
    var_12 = []
    var_13 = module_0.make_config(var_12, var_11)
    var_14 = '70'
    var_15 = '--exclude'
    var_16 = 'venv'
    var_17 = [var_3, var_14, var_15, var_16]
    var_18 = module_0.make_config(var_17, var_11)
    var_19 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_20 = module_1.loads(var_19)
    var_21 = []
    var_22 = module_0.make_config(var_21, var_20)
    var_23 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_24 = module_1.loads(var_23)
    var_25 = []
    var_26 = module_0.make_config(var_25, var_24)
    var_27 = []
    var_28 = module_0.make_config(var_27)
    var_29 = [var_5]
    var_30 = module_0.make_config(var_29, var_11)



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
    var_26 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_27 = '70'
    var_28 = [var_3, var_27, var_5]
    var_29 = 70
    var_30 = {var_10: var_29, var_11: var_14}
    var_31 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_32 = []
    var_33 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_34 = []
    var_35 = '--min-confidence'
    var_36 = '50'
    var_37 = [var_35, var_36]
    var_38 = module_0.make_config(var_37)



# Parsed testcases at query #33
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
    var_9 = '\n    [tool.vulture]\n    min_confidence = 75\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_10 = []
    var_11 = '60'
    var_12 = 'cli_path'
    var_13 = [var_3, var_11, var_12]
    var_14 = '\n    [tool.vulture]\n    invalid_key = "value"\n    '
    var_15 = []
    var_16 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_17 = []
    var_18 = []
    var_19 = module_0.make_config(var_18)
    var_20 = '\n    [tool.vulture]\n    paths = ["some_path"]\n    verbose = true\n    '
    var_21 = []
    var_22 = 'Reading configuration from'



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
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.make_config(var_6, var_1)
    var_8 = '\n[tool.vulture]\nmin_confidence = 30\nsort_by_size = true\n'
    var_9 = []
    var_10 = 0
    var_11 = [var_3, var_4]
    var_12 = '\n[tool.vulture]\ninvalid_key = 10\n'
    var_13 = []
    var_14 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_15 = []
    var_16 = []
    var_17 = None
    var_18 = module_0.make_config(var_16, var_17)
    var_19 = 'path1'
    var_20 = 'path2'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21, var_17)
    var_23 = '\n[tool.vulture]\npaths = ["path1", "path2"]\n'
    var_24 = []



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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nexclude = ["test_*.py"]\npaths = ["dir1", "dir2"]\n'
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
    var_10 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\npaths = ["toml_path1", "toml_path2"]\n'
    var_11 = []
    var_12 = '70'
    var_13 = 'cli_path1'
    var_14 = [var_3, var_12, var_5, var_13]
    var_15 = '\n[tool.vulture]\nunknown_key = "value"\n'
    var_16 = []
    var_17 = '\n[tool.vulture]\nmin_confidence = "not_an_int"\n'
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = []
    var_22 = module_0.make_config(var_21)



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
    var_19 = '--min-confidence'
    var_20 = '50'
    var_21 = [var_19, var_20]
    var_22 = module_0.make_config(var_21)



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
    var_6 = 'path1'
    var_7 = 'path2'
    var_8 = [var_3, var_4, var_5, var_6, var_7]
    var_9 = module_0.make_config(var_8)
    var_10 = '\n        [tool.vulture]\n        min_confidence = 30\n        verbose = true\n        paths = ["path3", "path4"]\n    '
    var_11 = []
    var_12 = 0
    var_13 = '60'
    var_14 = 'path5'
    var_15 = [var_3, var_13, var_5, var_14]
    var_16 = '--invalid-arg'
    var_17 = [var_16]
    var_18 = module_0.make_config(var_17)
    var_19 = '\n        [tool.vulture]\n        invalid_key = "value"\n    '
    var_20 = []
    var_21 = []
    var_22 = module_0.make_config(var_21)



# Parsed testcases at query #39
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
    var_12 = 'path5'
    var_13 = [var_3, var_4, var_12]
    var_14 = '[tool.vulture]\nunknown_key = 10'
    var_15 = []
    var_16 = []
    var_17 = '[tool.vulture]'
    var_18 = module_0.make_config(var_16, var_3)



# Parsed testcases at query #40
#--------------------------


import vulture.config as module_0

def test_case_0():
    var_0 = module_0.make_config()
    var_1 = '--min-confidence'
    var_2 = '50'
    var_3 = '--verbose'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.make_config(var_4)
    var_6 = '\n[tool.vulture]\nmin_confidence = 30\nverbose = true\n'
    var_7 = 0
    var_8 = [var_1, var_2]
    var_9 = '\n[tool.vulture]\ninvalid_key = "value"\n'
    var_10 = '--exclude'
    var_11 = 'test_*.py'
    var_12 = [var_10, var_11]
    var_13 = module_0.make_config(var_12)



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
    var_10 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = true\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_11 = []
    var_12 = '\n    [tool.vulture]\n    min_confidence = 75\n    verbose = false\n    paths = ["toml_path1"]\n    '
    var_13 = 'cli_path1'
    var_14 = [var_3, var_4, var_5, var_13]
    var_15 = '\n    [tool.vulture]\n    invalid_key = 123\n    '
    var_16 = []
    var_17 = '\n    [tool.vulture]\n    min_confidence = "not_an_int"\n    '
    var_18 = []
    var_19 = []
    var_20 = module_0.make_config(var_19)
    var_21 = [var_6, var_7]
    var_22 = module_0.make_config(var_21)
    var_23 = '\n    [tool.vulture]\n    paths = ["toml_path1", "toml_path2"]\n    '
    var_24 = '\n    [tool.vulture]\n    verbose = true\n    '
    var_25 = '--verbose'
    var_26 = [var_25]
    var_27 = 'Reading configuration from <_io.StringIO object>'



