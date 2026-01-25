####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = '# Comment\nimport b\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = '# isort: off\nimport b\nimport a\n# isort: on\n'
    var_7 = module_0.StringIO()
    var_8 = 'import c'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = module_0.StringIO()
    var_12 = True
    var_13 = [var_8]
    var_14 = module_1.Config()
    var_15 = ''
    var_16 = module_0.StringIO()
    var_17 = module_0.StringIO()
    var_18 = 'pyi'
    var_19 = '# isort: skip_file\nimport b\nimport a\n'
    var_20 = module_0.StringIO()
    var_21 = True
    var_22 = module_0.StringIO()
    var_23 = False
    var_24 = 'x = [3, 1, 2]\n'
    var_25 = module_0.StringIO()
    var_26 = module_1.Config()
    var_27 = "__all__ = ['b', 'a']\n"
    var_28 = module_0.StringIO()
    var_29 = module_1.Config()
    var_30 = module_0.StringIO()



# Parsed testcases at query #2
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = '# Comment\nimport b\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = 'x = 1\nimport b\nimport a\ny = 2\n'
    var_7 = module_0.StringIO()
    var_8 = 'import b\n# isort: off\nimport a\n'
    var_9 = module_0.StringIO()
    var_10 = 'import b\n# isort: split\nimport a\n'
    var_11 = module_0.StringIO()
    var_12 = 'import z'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = module_0.StringIO()
    var_16 = [var_12]
    var_17 = module_1.Config()
    var_18 = ''
    var_19 = module_0.StringIO()
    var_20 = True
    var_21 = module_1.Config()
    var_22 = module_0.StringIO()
    var_23 = '# isort: skip_file\nimport b\nimport a\n'
    var_24 = module_0.StringIO()
    var_25 = module_0.StringIO()
    var_26 = False
    var_27 = 'x = [3, 1, 2]\n# isort: code\n'
    var_28 = module_0.StringIO()
    var_29 = module_1.Config()
    var_30 = "__all__ = ['c', 'a', 'b']\n"
    var_31 = module_0.StringIO()
    var_32 = 'cimport b\ncimport a\n'
    var_33 = module_0.StringIO()
    var_34 = 'pyx'
    var_35 = module_1.Config()
    var_36 = module_0.StringIO()
    var_37 = module_1.Config()
    var_38 = module_0.StringIO()
    var_39 = '\r\n'
    var_40 = module_1.Config()
    var_41 = 'import b\r\nimport a\r\n'
    var_42 = module_0.StringIO()
    var_43 = module_1.Config()
    var_44 = module_0.StringIO()
    var_45 = '# Comment'
    var_46 = [var_45]
    var_47 = module_1.Config()
    var_48 = module_0.StringIO()
    var_49 = 2
    var_50 = module_1.Config()
    var_51 = '\n\nimport b\nimport a\n'
    var_52 = module_0.StringIO()
    var_53 = module_1.Config()
    var_54 = module_0.StringIO()
    var_55 = module_1.Config()
    var_56 = module_0.StringIO()
    var_57 = '# Section'
    var_58 = [var_57]
    var_59 = module_1.Config()
    var_60 = '# Section\nimport b\nimport a\n'
    var_61 = module_0.StringIO()
    var_62 = '# End'
    var_63 = [var_62]
    var_64 = module_1.Config()
    var_65 = 'import b\nimport a\n# End\n'
    var_66 = module_0.StringIO()
    var_67 = '    import b\n    import a\n'
    var_68 = module_0.StringIO()
    var_69 = 'yield\nimport b\nimport a\n'
    var_70 = module_0.StringIO()
    var_71 = 'raise\nimport b\nimport a\n'
    var_72 = module_0.StringIO()
    var_73 = '"""Docstring"""'
    var_74 = module_0.StringIO()
    var_75 = module_0.StringIO()
    var_76 = '   \n   \n'
    var_77 = module_0.StringIO()



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = '# Comment\nimport b\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = 'import b\n# isort: off\nimport a\n'
    var_7 = module_0.StringIO()
    var_8 = 'import b\n# isort: split\nimport a\n'
    var_9 = module_0.StringIO()
    var_10 = 'from __future__ import annotations'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'import b\n'
    var_14 = module_0.StringIO()
    var_15 = True
    var_16 = module_1.Config()
    var_17 = ''
    var_18 = module_0.StringIO()
    var_19 = '# isort: skip_file\nimport b\nimport a\n'
    var_20 = module_0.StringIO()
    var_21 = 'x = [3, 1, 2]\n# isort: literal\n'
    var_22 = module_0.StringIO()
    var_23 = module_1.Config()
    var_24 = "__all__ = ['c', 'b', 'a']\n"
    var_25 = module_0.StringIO()



# Parsed testcases at query #2
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = 'x = 1\nimport b\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = '# comment\nimport b\nimport a\n'
    var_7 = module_0.StringIO()
    var_8 = '# isort: off\nimport b\nimport a\n# isort: on\n'
    var_9 = module_0.StringIO()
    var_10 = 'from __future__ import annotations'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = module_0.StringIO()
    var_14 = ''
    var_15 = module_0.StringIO()
    var_16 = '# comment\n# another comment\n'
    var_17 = module_0.StringIO()
    var_18 = 'x = {3, 2, 1}\n'
    var_19 = module_0.StringIO()
    var_20 = "__all__ = ['c', 'b', 'a']\n"
    var_21 = module_0.StringIO()
    var_22 = True
    var_23 = module_1.Config()



# Parsed testcases at query #3
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = '# comment\nimport b\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = 'x = 1\nimport b\nimport a\ny = 2\n'
    var_7 = module_0.StringIO()
    var_8 = '# isort: off\nimport b\nimport a\n# isort: on\n'
    var_9 = module_0.StringIO()
    var_10 = 'import z'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'import a\n'
    var_14 = module_0.StringIO()
    var_15 = True
    var_16 = module_1.Config()
    var_17 = ''
    var_18 = module_0.StringIO()
    var_19 = '# isort: skip_file\nimport b\nimport a\n'
    var_20 = module_0.StringIO()
    var_21 = 'x = [3, 1, 2]\n# isort: code\n'
    var_22 = module_0.StringIO()
    var_23 = module_1.Config()
    var_24 = "__all__ = ['c', 'b', 'a']\n"
    var_25 = module_0.StringIO()



# Parsed testcases at query #4
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = 'import z'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'import a\n'
    var_8 = module_0.StringIO()
    var_9 = '# isort: off\nimport b\nimport a\n# isort: on\nimport c\n'
    var_10 = module_0.StringIO()
    var_11 = '# isort: skip_file\nimport b\nimport a\n'
    var_12 = module_0.StringIO()
    var_13 = "# isort: list\n['c', 'a', 'b']\n"
    var_14 = module_0.StringIO()
    var_15 = "__all__ = ['b', 'a']\n"
    var_16 = module_0.StringIO()
    var_17 = 'cimport b\ncimport a\n'
    var_18 = module_0.StringIO()
    var_19 = 'pyx'
    var_20 = 'import b\nx = 1\nimport a\n'
    var_21 = module_0.StringIO()
    var_22 = ''
    var_23 = module_0.StringIO()
    var_24 = '# comment\n# another comment\n'
    var_25 = module_0.StringIO()



# Parsed testcases at query #5
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = 'import b\nx = 1\nimport a\n'
    var_5 = module_0.StringIO()
    var_6 = '# comment\nimport b\nimport a\n'
    var_7 = module_0.StringIO()
    var_8 = '# isort: off\nimport b\nimport a\n# isort: on\nimport c\n'
    var_9 = module_0.StringIO()
    var_10 = 'import z'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'import a\n'
    var_14 = module_0.StringIO()
    var_15 = ''
    var_16 = module_0.StringIO()
    var_17 = '# comment\n# another comment\n'
    var_18 = module_0.StringIO()
    var_19 = 'x = [3, 1, 2]  # isort: sort\n'
    var_20 = module_0.StringIO()
    var_21 = "__all__ = ['b', 'a']\n"
    var_22 = module_0.StringIO()



