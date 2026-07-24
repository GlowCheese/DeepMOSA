####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = False



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = 'test.py'
    var_4 = module_1.Path(var_3)
    var_5 = True



# Parsed testcases at query #4
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = True
    var_3 = '/non/existent/path'
    var_4 = [var_3]
    var_5 = module_0.find_imports_in_paths(var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = list(var_0)
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import zipfile as module_1
import _io as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = "print('Hello, world!')\n"
    var_3 = '# isort:skip_file\nimport sys\nimport os\n'
    var_4 = 'test_file.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = 'test_file.py'
    var_8 = module_1.Path(var_7)
    var_9 = "import sys\nimport os\nprint('Hello, world!'"
    var_10 = module_2.StringIO()
    var_11 = 'py'
    var_12 = True
    var_13 = 'black'



# Parsed testcases at query #6
#--------------------------


import _io as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = module_1.Config()
    var_3 = 0
    var_4 = 'import a\nimport b\n'
    var_5 = module_0.StringIO()
    var_6 = module_0.StringIO()
    var_7 = True
    var_8 = module_0.StringIO()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'example_path'
    var_1 = [var_0]



# Parsed testcases at query #8
#--------------------------


import zipfile as module_0
import isort.api as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'test_correct.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.check_file(var_1)
    assert var_3 is True
    var_4 = 'test_incorrect.py'
    var_5 = module_0.Path(var_4)
    var_6 = 'import sys\nimport os\n'
    var_7 = module_1.check_file(var_5)
    assert var_7 is False
    var_8 = 'test_skip.py'
    var_9 = module_0.Path(var_8)
    var_10 = '# isort: skip_file\nimport sys\nimport os\n'
    var_11 = module_1.check_file(var_9)
    var_12 = 'test_syntax_error.py'
    var_13 = module_0.Path(var_12)
    var_14 = 'import sys\nimport os\nasdf\n'
    var_15 = module_1.check_file(var_13)
    var_16 = 'test_introduced_syntax_error.py'
    var_17 = module_0.Path(var_16)
    var_18 = True
    var_19 = module_2.Config()
    var_20 = module_1.check_file(var_17, config=var_19)
    var_21 = 'test_show_diff.py'
    var_22 = module_0.Path(var_21)
    var_23 = True
    var_24 = module_1.check_file(var_22, var_23)
    assert var_24 is True
    var_25 = 'test_disregard_skip.py'
    var_26 = module_0.Path(var_25)
    var_27 = module_1.check_file(var_26, disregard_skip=var_23)
    assert var_27 is False



# Parsed testcases at query #9
#--------------------------


import isort.api as module_0
import isort.settings as module_1
import _io as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 0
    var_2 = "import os\nimport sys\nprint('hello')\n"
    var_3 = 0
    var_4 = "print('hello')\n"
    var_5 = 0
    var_6 = '# isort: skip_file\nimport os\nimport sys\n'
    var_7 = 0
    var_8 = module_0.sort_file(var_6)
    var_9 = 'import os\nimport sys\n'
    var_10 = 0
    var_11 = '*.py'
    var_12 = [var_11]
    var_13 = module_1.Config()
    var_14 = module_0.sort_file(var_9, config=var_13)
    var_15 = 'import os\nimport sys\n'
    var_16 = 0
    var_17 = '*.py'
    var_18 = [var_17]
    var_19 = module_1.Config()
    var_20 = True
    var_21 = '# isort: skip_file\nimport os\nimport sys\n'
    var_22 = 0
    var_23 = True
    var_24 = module_0.sort_file(var_17, disregard_skip=var_23)
    assert var_24 is True
    var_25 = '# isort: skip_file\nimport os\nimport sys\n'
    var_26 = 0
    var_27 = True
    var_28 = module_0.sort_file(var_25, show_diff=var_27)
    var_29 = 'import os\nimport sys\n'
    var_30 = 0
    var_31 = '*.py'
    var_32 = [var_31]
    var_33 = module_1.Config()
    var_34 = True
    var_35 = module_0.sort_file(var_29, config=var_33, show_diff=var_34)
    var_36 = 'import os\nimport sys\n'
    var_37 = 0
    var_38 = '*.py'
    var_39 = [var_38]
    var_40 = module_1.Config()
    var_41 = True
    var_42 = module_0.sort_file(var_24, config=var_40, disregard_skip=var_41, show_diff=var_41)
    assert var_42 is True
    assert var_42 == '# isort: skip_file\nimport os\nimport sys\n'
    var_43 = '# isort: skip_file\nimport os\nimport sys\n'
    var_44 = 0
    var_45 = True
    var_46 = module_0.sort_file(var_38, disregard_skip=var_45, show_diff=var_45)
    assert var_46 is True
    var_47 = '# isort: skip_file\nimport os\nimport sys\n'
    var_48 = 0
    var_49 = True
    var_50 = module_0.sort_file(var_47, write_to_stdout=var_49)
    var_51 = 'import os\nimport sys\n'
    var_52 = 0
    var_53 = '*.py'
    var_54 = [var_53]
    var_55 = module_1.Config()
    var_56 = True
    var_57 = module_0.sort_file(var_51, config=var_55, write_to_stdout=var_56)
    var_58 = 'import os\nimport sys\n'
    var_59 = 0
    var_60 = '*.py'
    var_61 = [var_60]
    var_62 = module_1.Config()
    var_63 = True
    var_64 = module_0.sort_file(var_46, config=var_62, disregard_skip=var_63, write_to_stdout=var_63)
    assert var_64 is True
    assert var_64 == '# isort: skip_file\nimport os\nimport sys\n'
    var_65 = '# isort: skip_file\nimport os\nimport sys\n'
    var_66 = 0
    var_67 = True
    var_68 = module_0.sort_file(var_60, disregard_skip=var_67, write_to_stdout=var_67)
    assert var_68 is True
    var_69 = '# isort: skip_file\nimport os\nimport sys\n'
    var_70 = 0
    var_71 = module_2.StringIO()
    var_72 = module_0.sort_file(var_69, output=var_71)
    var_73 = 'import os\nimport sys\n'
    var_74 = 0
    var_75 = '*.py'
    var_76 = [var_75]
    var_77 = module_1.Config()
    var_78 = module_2.StringIO()
    var_79 = module_0.sort_file(var_73, config=var_77, output=var_78)
    var_80 = 'import os\nimport sys\n'
    var_81 = 0
    var_82 = '*.py'
    var_83 = [var_82]
    var_84 = module_1.Config()



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'import os\nfrom sys import path\nimport numpy as np\n'
    var_1 = True
    var_2 = 'import os\nfrom sys import path\nimport numpy as np\ndef foo():\n    pass\n'



# Parsed testcases at query #11
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'test_file1.py'
    var_2 = module_0.check_file(var_1)
    assert var_2 is True
    var_3 = 'import sys\nimport os\n'
    var_4 = 'test_file2.py'
    var_5 = module_0.check_file(var_4)
    assert var_5 is False
    var_6 = 'non_existent_file.py'
    var_7 = module_0.check_file(var_6)
    var_8 = '# isort:skip_file\nimport sys\nimport os\n'
    var_9 = 'test_file3.py'
    var_10 = module_0.check_file(var_9)
    var_11 = 'import sys\nimport os\n'
    var_12 = 'test_file4.py'
    var_13 = True
    var_14 = module_0.check_file(var_12, var_13)
    assert var_14 is False



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'test2.py'
    var_2 = 'import os\nimport sys\n'
    var_3 = 'from datetime import datetime\n'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = 'datetime'
    var_7 = 'import os\nimport os\n'
    var_8 = True
    var_9 = 'import os\ndef foo():\n    import sys\n'



# Parsed testcases at query #13
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.StringIO()
    var_2 = 0
    var_3 = 'import sys\nimport os'
    var_4 = module_0.StringIO()



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = False
    var_2 = 'import a\nimport b\n'



# Parsed testcases at query #15
#--------------------------


import zipfile as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = 'import os\nimport os.path'
    var_2 = True
    var_3 = 'import os\ndef foo():\n    import sys'
    var_4 = 'import os'
    var_5 = 'test.py'
    var_6 = module_0.Path(var_5)
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = module_1.Config()
    var_10 = set()
    var_11 = 'All tests passed for find_imports_in_stream'
    var_12 = print(var_11)



# Parsed testcases at query #16
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 0
    var_3 = 'import a\nimport b\n'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = 'py'
    var_7 = 'test.py'
    var_8 = module_1.Path(var_7)
    var_9 = module_0.StringIO()
    var_10 = [var_7]
    var_11 = module_2.Config()
    var_12 = 'test.py'
    var_13 = module_1.Path(var_12)
    var_14 = module_0.StringIO()
    var_15 = [var_7]
    var_16 = module_2.Config()
    var_17 = module_1.Path(var_7)
    var_18 = True
    var_19 = module_0.StringIO()
    var_20 = module_0.StringIO()
    var_21 = 'import b\nimport a\nx = '
    var_22 = module_0.StringIO()
    var_23 = module_2.Config()
    var_24 = 'import b\nimport a\nx = 1'
    var_25 = module_0.StringIO()
    var_26 = module_2.Config()
    var_27 = '# isort: skip_file\nimport b\nimport a\n'
    var_28 = module_0.StringIO()
    var_29 = 'All test cases passed!'
    var_30 = print(var_29)



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nimport math\n'
    var_1 = False
    var_2 = 'import math\nimport os\nimport sys\n'
    var_3 = False
    var_4 = 'import sys\nimport os\nimport math\n'
    var_5 = True
    var_6 = 'import sys\nimport os\nimport math\n'
    var_7 = True
    var_8 = 'import sys\nimport os\nimport math\n'
    var_9 = module_0.Config()
    var_10 = 'All test cases passed!'
    var_11 = print(var_10)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = "print('Hello, world!')\n"
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'import os\n'
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = 'import os\nimport sys\n'
    var_6 = len(var_1)
    assert var_6 == 2



# Parsed testcases at query #19
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'test_file1.py'
    var_2 = module_0.check_file(var_1)
    assert var_2 is True
    var_3 = 'import sys\nimport os\n'
    var_4 = 'test_file2.py'
    var_5 = module_0.check_file(var_4)
    assert var_5 is False
    var_6 = '# isort: skip_file\nimport sys\nimport os\n'
    var_7 = 'test_file3.py'
    var_8 = module_0.check_file(var_7)
    var_9 = 'non_existent_file.py'
    var_10 = module_0.check_file(var_9)
    var_11 = 'test_file3.py'



# Parsed testcases at query #20
#--------------------------


import zipfile as module_0
import isort.settings as module_1
import _io as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = 'test.py'
    var_4 = module_0.Path(var_3)
    var_5 = 'py'
    var_6 = 'import sys\nimport os\n'
    var_7 = [var_3]
    var_8 = module_1.Config()
    var_9 = module_0.Path(var_3)
    var_10 = True
    var_11 = 'import sys\nimport os\n'
    var_12 = module_2.StringIO()
    var_13 = 0
    var_14 = ''
    var_15 = '# This is a comment\n'
    var_16 = 'import sys\nimport os\nx ='
    var_17 = 'import sys\nimport os\nx ='
    var_18 = False
    var_19 = module_1.Config()
    var_20 = 'pyx'
    var_21 = 'import sys\nimport os\nx ='
    var_22 = module_2.StringIO()



# Parsed testcases at query #21
#--------------------------


import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = 'test_file.py'
    var_4 = module_1.Path(var_3)
    var_5 = True
    var_6 = 'line_length'
    var_7 = 80
    var_8 = {var_6: var_7}



# Parsed testcases at query #22
#--------------------------


import zipfile as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import os\nimport sys'
    var_2 = 'import os\nimport os.path'
    var_3 = True
    var_4 = 'import os\ndef foo():\n    import sys'
    var_5 = 'test.py'
    var_6 = module_0.Path(var_5)
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = module_1.Config()
    var_10 = {var_7}
    var_11 = 'import os as operating_system\nimport os'
    var_12 = 'from os import path\nfrom os import path'
    var_13 = 'from os import path\nimport os'
    var_14 = 'from os.path import join\nimport os'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = True
    var_2 = module_0.StringIO()
    var_3 = 0
    var_4 = 'import a\nimport b\n'
    var_5 = 'import b\nimport a\nsyntax error\n'
    var_6 = 'import b\nimport a\n'
    var_7 = 'n'
    assert var_7 == 'import b\nimport a\n'
    assert var_7 == 'import a\nimport b\n'
    var_8 = True
    var_9 = 'y'



# Parsed testcases at query #2
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 0
    var_2 = 'import os\nimport sys\n'
    var_3 = 0
    var_4 = "print('Hello, World!')\n"
    var_5 = 0
    var_6 = "import os\nprint('Hello, World!'\n"
    var_7 = 0
    var_8 = module_0.find_imports_in_file(var_6)
    var_9 = list(var_8)
    var_10 = 'import os\nimport os\n'
    var_11 = 0
    var_12 = True
    var_13 = len(var_9)
    assert var_13 == 1
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = 0
    var_16 = True
    var_17 = len(var_9)
    assert var_17 == 1
    var_18 = 'import os\nimport os.path\n'
    var_19 = 0
    var_20 = len(var_9)
    assert var_20 == 1
    var_21 = 'import os\nimport os.path\nimport sys\n'
    var_22 = 0
    var_23 = len(var_9)
    assert var_23 == 2
    var_24 = 'from os import path\nfrom os import path\n'
    var_25 = 0
    var_26 = len(var_9)
    assert var_26 == 1
    var_27 = 'import os as operating_system\nimport os as operating_system\n'
    var_28 = 0
    var_29 = len(var_9)
    assert var_29 == 1



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom collections import defaultdict\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'collections'
    var_6 = 'defaultdict'
    var_7 = True
    var_8 = len(var_1)
    assert var_8 == 3
    var_9 = len(var_1)
    assert var_9 == 3
    var_10 = len(var_1)
    assert var_10 == 3
    var_11 = [var_5]
    var_12 = module_0.Config()
    var_13 = len(var_1)
    assert var_13 == 3



# Parsed testcases at query #4
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = False
    var_2 = module_0.StringIO()
    var_3 = True
    var_4 = False
    var_5 = module_0.StringIO()
    var_6 = 'import b\nimport a\n'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = "print('Hello, World!')"
    var_1 = 'import os'
    var_2 = 'import os\nimport sys'
    var_3 = 'import os\nimport os\nimport sys'
    var_4 = True
    var_5 = 'import os\ndef foo():\n    import sys'
    var_6 = 'import os\nfrom os import path\nimport sys'
    var_7 = 'import os.path\nimport os\nimport sys'



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'test.py'
    var_2 = True
    var_3 = module_0.sort_file(var_1, write_to_stdout=var_2)
    assert var_3 is True
    var_4 = 'import sys\nimport os\n'
    var_5 = module_0.sort_file(var_4, write_to_stdout=var_2)
    assert var_5 is True
    var_6 = 'import os\nimport os\n'
    var_7 = module_0.sort_file(var_6, write_to_stdout=var_2)
    assert var_7 is True
    var_8 = 'import sys\nimport os\nfrom math import pi\n'
    var_9 = module_0.sort_file(var_8, write_to_stdout=var_2)
    assert var_9 is True
    var_10 = "print('Hello, World!')\n"
    var_11 = module_0.sort_file(var_10, write_to_stdout=var_2)
    assert var_11 is False
    var_12 = "import os\nprint('Hello, World!'\n"
    var_13 = 'test.py'
    var_14 = True
    var_15 = module_0.sort_file(var_13, write_to_stdout=var_14)
    var_16 = False
    assert var_16 is False
    var_17 = 'import os\n# This is a comment\nimport sys\n'
    var_18 = module_0.sort_file(var_17, write_to_stdout=var_14)
    assert var_18 is True
    var_19 = 'import os\n"""This is a docstring"""\nimport sys\n'
    var_20 = module_0.sort_file(var_19, write_to_stdout=var_14)
    assert var_20 is True
    var_21 = '#!/usr/bin/env python\nimport os\nimport sys\n'
    var_22 = module_0.sort_file(var_21, write_to_stdout=var_14)
    assert var_22 is True
    var_23 = 'import os\nimport sys\n'
    var_24 = module_0.sort_file(var_23, write_to_stdout=var_14)
    assert var_24 is True



# Parsed testcases at query #8
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = '\nimport os\nimport sys\nfrom typing import List, Dict\n'
    var_1 = module_0.find_imports_in_code(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 3



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = True

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = True

import _io as module_0

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = module_0.StringIO()
    var_2 = 'import os'
    var_3 = 'import sys'

import _io as module_0

def test_case_0():
    var_0 = 'import sys\nimport os\n'
    var_1 = module_0.StringIO()
    var_2 = 'import os'
    var_3 = 'import sys'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = 'py'
    var_4 = 'black'
    var_5 = module_0.Config()
    var_6 = 'test.py'
    var_7 = module_1.Path(var_6)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\nimport b\nimport a\n'
    var_1 = False
    var_2 = '\nimport a\nimport b\n'



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = module_0.find_imports_in_paths(var_0)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nimport math\n'
    var_1 = True



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import os\nimport sys\n'
    var_3 = True
    var_4 = 'import sys\nimport os\n'



# Parsed testcases at query #17
#--------------------------


import zipfile as module_0
import isort.settings as module_1
import isort.api as module_2

def test_case_0():
    var_0 = 'test1.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'test2.py'
    var_3 = module_0.Path(var_2)
    var_4 = [var_1, var_3]
    var_5 = module_1.Config()
    var_6 = 'force_to_top'
    var_7 = True
    var_8 = {var_6: var_7}
    var_9 = module_2.find_imports_in_paths(var_4, var_5, **var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #18
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = 'os'
    var_2 = None
    var_3 = 1
    var_4 = module_0.Import()
    var_5 = [var_4]
    var_6 = 'import sys as s'
    var_7 = 'sys'
    var_8 = 's'
    var_9 = module_0.Import()
    var_10 = [var_9]
    var_11 = 'from os import path'
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = [var_13]
    var_15 = 'import os, sys'
    var_16 = module_0.Import()
    var_17 = module_0.Import()
    var_18 = [var_16, var_17]
    var_19 = 'import os\nimport os'
    var_20 = True
    var_21 = module_0.Import()
    var_22 = [var_21]
    var_23 = 'import os\ndef func(): pass'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = [var_25]
    var_27 = 'import os\nfrom sys import path'
    var_28 = module_0.Import()
    var_29 = 'from sys import path'
    var_30 = 2
    var_31 = module_0.Import()
    var_32 = [var_28, var_31]



# Parsed testcases at query #19
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = 'pathlib'
    var_6 = 'Path'
    var_7 = True
    var_8 = len(var_1)
    assert var_8 == 3
    var_9 = len(var_1)
    assert var_9 == 3
    var_10 = len(var_1)
    assert var_10 == 3
    var_11 = 'import os\ndef foo():\n    import sys\n'
    var_12 = module_0.Path(var_3)
    var_13 = module_1.find_imports_in_file(var_12, top_only=var_7)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = True
    var_17 = True



# Parsed testcases at query #20
#--------------------------


import zipfile as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import os\nimport sys'
    var_2 = 'import os\nimport os'
    var_3 = True
    var_4 = 'import os\ndef foo():\n    import sys'
    var_5 = 'test.py'
    var_6 = module_0.Path(var_5)
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = module_1.Config()



# Parsed testcases at query #21
#--------------------------


import _io as module_0
import isort.api as module_1
import isort.settings as module_2
import zipfile as module_3

def test_case_0():
    var_0 = module_0.StringIO()
    var_1 = module_0.StringIO()
    var_2 = module_1.sort_stream(var_0, var_1)
    assert var_2 is False
    var_3 = 'import os\nimport sys\n'
    var_4 = module_0.StringIO()
    var_5 = module_1.sort_stream(var_0, var_4)
    assert var_5 is False
    var_6 = 'import sys\nimport os\n'
    var_7 = module_0.StringIO()
    var_8 = module_1.sort_stream(var_0, var_7)
    assert var_8 is True
    var_9 = 0
    var_10 = '# isort:skip_file\nimport sys\nimport os\n'
    var_11 = module_0.StringIO()
    var_12 = module_1.sort_stream(var_0, var_11)
    var_13 = 'test_file.py'
    var_14 = [var_13]
    var_15 = module_2.Config()
    var_16 = module_0.StringIO()
    var_17 = 'test_file.py'
    var_18 = module_3.Path(var_17)
    var_19 = module_1.sort_stream(var_0, var_16, config=var_15, file_path=var_18)
    var_20 = 'import sys\nimport os\ninvalid syntax\n'
    var_21 = module_0.StringIO()
    var_22 = module_1.sort_stream(var_0, var_21)



# Parsed testcases at query #22
#--------------------------


import isort.api as module_0
import _io as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'import b\nimport a\n'
    var_2 = module_0.sort_file(var_0)
    assert var_2 is True
    assert var_2 == 'import a\nimport b\n'
    var_3 = 'test_file.py'
    var_4 = 'import a\nimport b\n'
    assert var_4 == 'import a\nimport b\n'
    var_5 = module_0.sort_file(var_3)
    assert var_5 is False
    var_6 = 'test_file.py'
    var_7 = '# isort: skip_file\nimport b\nimport a\n'
    assert var_7 == '# isort: skip_file\nimport b\nimport a\n'
    var_8 = module_0.sort_file(var_6)
    assert var_8 is False
    var_9 = 'test_file.py'
    var_10 = '# isort: skip_file\nimport b\nimport a\n'
    assert var_10 == '# isort: skip_file\nimport a\nimport b\n'
    var_11 = True
    var_12 = module_0.sort_file(var_9, disregard_skip=var_11)
    assert var_12 is True
    var_13 = 'test_file.py'
    var_14 = 'import b\nimport a\nSyntaxError\n'
    assert var_14 == 'import b\nimport a\nSyntaxError\n'
    var_15 = module_0.sort_file(var_13)
    assert var_15 is False
    var_16 = 'test_file.py'
    var_17 = 'import b\nimport a\n'
    assert var_17 == 'import a\nimport b\n'
    var_18 = module_0.sort_file(var_16, write_to_stdout=var_11)
    assert var_18 is True
    var_19 = 'test_file.py'
    var_20 = module_1.StringIO()
    var_21 = 'import b\nimport a\n'
    var_22 = module_0.sort_file(var_19, output=var_20)
    assert var_22 is True
    var_23 = 0
    var_24 = 'test_file.py'
    var_25 = module_1.StringIO()
    var_26 = 'import b\nimport a\n'
    var_27 = module_0.sort_file(var_24, show_diff=var_25)
    assert var_27 is True
    var_28 = 'test_file.py'
    var_29 = 'import b\nimport a\n'
    assert var_29 == 'import a\nimport b\n'
    var_30 = module_0.sort_file(var_28, ask_to_apply=var_11)
    assert var_30 is True



# Parsed testcases at query #23
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = 'import a\nimport b\n'
    var_2 = module_0.StringIO()
    var_3 = module_0.StringIO()



# Parsed testcases at query #24
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'import b\nimport a\n'
    var_2 = module_0.sort_file(var_0)
    assert var_2 is True
    assert var_2 == 'import a\nimport b\n'
    var_3 = 'test_file.py'
    var_4 = 'import a\nimport b\n'
    var_5 = module_0.sort_file(var_3)
    assert var_5 is False
    var_6 = 'test_file.py'
    var_7 = 'import b\nimport a\n'
    var_8 = 'test_file.py'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = module_0.sort_file(var_6, config=var_10)
    assert var_11 is False
    var_12 = 'test_file.py'
    var_13 = 'import b\nimport a\ninvalid syntax\n'
    var_14 = True
    var_15 = module_1.Config()
    var_16 = module_0.sort_file(var_12, config=var_15)
    assert var_16 is False
    var_17 = 'test_file.py'
    var_18 = 'import b\nimport a\n'
    assert var_18 == 'import a\nimport b\n'
    var_19 = module_1.Config()
    var_20 = module_0.sort_file(var_17, config=var_19)
    assert var_20 is True



# Parsed testcases at query #25
#--------------------------




