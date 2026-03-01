####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 79
    var_9 = module_1.Config()
    var_10 = 'import os\nimport sys\n'



# Parsed testcases at query #2
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'test1.py'
    var_3 = 'import sys\nimport os\n'
    var_4 = 'test2.py'
    var_5 = 'from typing import List\nimport json\n'
    var_6 = 'test.py'
    var_7 = 'import sys\nimport sys\n'
    var_8 = True
    var_9 = list(var_5)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'test.py'
    var_12 = 'import sys\n\ndef foo():\n    import os\n'
    var_13 = True
    var_14 = list(var_5)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'nonexistent.py'
    var_17 = module_0.find_imports_in_paths(var_12)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0



# Parsed testcases at query #3
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path\nimport sys\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'import os\n\ndef foo():\n    import sys\n'
    var_4 = True
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_2 = 'import os\nimport os\nfrom os import path\n'
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent.py'
    var_6 = list(var_0)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 79
    var_9 = module_0.Config()
    var_10 = len(var_6)
    assert var_10 == 3



# Parsed testcases at query #5
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\ndef foo():\n    import sys\n'
    var_5 = True
    var_6 = list(var_2)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'non_existent_file.py'
    var_9 = module_0.find_imports_in_file(var_8)
    var_10 = list(var_9)



# Parsed testcases at query #6
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
    var_11 = False
    var_12 = module_1.check_file(var_9, disregard_skip=var_11)
    assert var_12 is True
    var_13 = 'non_existent.py'
    var_14 = module_0.Path(var_13)
    var_15 = module_1.check_file(var_14)
    var_16 = 'test_syntax_error.py'
    var_17 = module_0.Path(var_16)
    var_18 = 'import os\nimport\n'
    var_19 = module_1.check_file(var_17)
    var_20 = 'test_custom_config.py'
    var_21 = module_0.Path(var_20)
    var_22 = 79
    var_23 = module_2.Config()
    var_24 = module_1.check_file(var_21, config=var_23)
    assert var_24 is False



# Parsed testcases at query #7
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import sys\ndef foo():\n    import os\n'
    var_5 = True
    var_6 = list(var_2)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'non_existent_file.py'
    var_9 = module_0.find_imports_in_file(var_8)
    var_10 = list(var_9)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom os import path\nfrom os import path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom os import path\nfrom os import path'
    var_4 = 'from os import path\nfrom os import path\nfrom os import listdir'
    var_5 = 'import os\ndef foo():\n    import sys'
    var_6 = ''
    var_7 = 'import os\nimport sys'
    var_8 = 'os'
    var_9 = [var_8]
    var_10 = module_0.Config()



# Parsed testcases at query #9
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.find_imports_in_paths(var_0)
    var_2 = list(var_1)
    var_3 = 'non_existent_path'
    var_4 = [var_3]
    var_5 = module_0.find_imports_in_paths(var_4)
    var_6 = list(var_5)
    var_7 = 'import os\nimport sys\nfrom pathlib import Path'
    var_8 = [var_3]
    var_9 = module_0.find_imports_in_paths(var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = 'file1.py'
    var_13 = var_7 / var_12
    var_14 = 'file2.py'
    var_15 = var_2 / var_14
    var_16 = 'import json\nfrom typing import List'
    var_17 = 'import re\nfrom collections import defaultdict'
    var_18 = [var_13, var_15]
    var_19 = module_0.find_imports_in_paths(var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 'file1.py'
    var_23 = var_7 / var_22
    var_24 = 'file2.py'
    var_25 = var_2 / var_24
    var_26 = 'import os\nimport sys\nimport os'
    var_27 = 'import sys\nimport json'
    var_28 = [var_23, var_25]
    var_29 = True
    var_30 = module_0.find_imports_in_paths(var_28, unique=var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 3
    var_33 = 'import os\n\ndef foo():\n    import sys\n'
    var_34 = [var_24]
    var_35 = True
    var_36 = module_0.find_imports_in_paths(var_34, top_only=var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 0
    var_5 = 'from os import path\nfrom os import path as os_path'
    var_6 = 'import os\nimport os.path\nfrom os import path'
    var_7 = 'import os.path\nimport os.sys\nfrom os import path'
    var_8 = 'import os\n\ndef foo():\n    import sys\n\nimport pathlib'
    var_9 = 'import os\nimport sys'
    var_10 = module_0.Config()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_1 = 'import json\nimport os\nimport sys\n\nfrom pathlib import Path\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_3 = 0
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_5 = True
    var_6 = 0



# Parsed testcases at query #12
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path\nimport sys\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)
    var_8 = True
    var_9 = True



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_5 = True
    var_6 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_7 = True
    var_8 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_9 = True



# Parsed testcases at query #14
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'import sys\nimport os\ninvalid syntax here\n'
    var_9 = module_0.check_file(var_8)
    var_10 = 'import sys\nimport os\n'
    var_11 = 79
    var_12 = module_1.Config()



# Parsed testcases at query #15
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.sort_file(var_1)
    assert var_3 is True
    var_4 = var_1.read_text()
    assert var_4 == 'import os\nimport sys\n'



# Parsed testcases at query #16
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\n\nfrom pathlib import Path\n'
    var_2 = 'import sys\nimport os\n\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import _io as module_1
import isort.api as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = '# isort: skip_file\nimport sys\nimport os\n'
    var_3 = True
    var_4 = 'import sys\nimport os\n'
    var_5 = 'test.py'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = True
    var_9 = 'import sys\nimport os\n'
    var_10 = module_1.StringIO()
    var_11 = module_2.check_file(var_5, var_10)
    assert var_11 is False
    var_12 = 'non_existent_file.py'
    var_13 = module_2.check_file(var_12)



# Parsed testcases at query #2
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_5 = module_0.StringIO()
    var_6 = True
    var_7 = 0



# Parsed testcases at query #3
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.find_imports_in_code(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'from collections import defaultdict\nfrom typing import List'
    var_5 = module_0.find_imports_in_code(var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'import os\nfrom sys import argv\nimport json'
    var_9 = module_0.find_imports_in_code(var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = 'import os\nimport os'
    var_13 = True
    var_14 = module_0.find_imports_in_code(var_12, unique=var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'import os\ndef foo():\n    import sys'
    var_18 = module_0.find_imports_in_code(var_17, top_only=var_13)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = ''
    var_22 = module_0.find_imports_in_code(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = 'x = 1\ny = 2'
    var_26 = module_0.find_imports_in_code(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = 'import os\nimport sys'
    var_30 = 'force_single_line'
    var_31 = {var_30: var_13}
    var_32 = module_0.find_imports_in_code(var_29, **var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2



# Parsed testcases at query #4
#--------------------------


import isort.api as module_0
import _io as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = module_0.sort_file(var_2, show_diff=var_3)
    var_5 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_6 = True
    var_7 = module_0.sort_file(var_5, write_to_stdout=var_6)
    var_8 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_9 = module_1.StringIO()
    var_10 = 0
    var_11 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_12 = True
    var_13 = module_0.sort_file(var_11, ask_to_apply=var_12)
    assert var_13 is False
    var_14 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_15 = True
    var_16 = module_0.sort_file(var_14, ask_to_apply=var_15)
    assert var_16 is True
    var_17 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_18 = True
    var_19 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_20 = 50
    var_21 = 'import os\nimport sys\nfrom pathlib import Path\ninvalid syntax\n'
    var_22 = module_0.sort_file(var_21)
    assert var_22 is False



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #7
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'import sys\nimport os\nthis is not valid python'
    var_9 = module_0.check_file(var_8)
    var_10 = 'import sys\nimport os\n'
    var_11 = 79
    var_12 = module_1.Config()



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom os import path\nfrom os import path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom os import path\nfrom os import path'
    var_4 = 'from os import path\nfrom os import path\nfrom os import listdir'
    var_5 = 'import os\n\ndef foo():\n    import sys'
    var_6 = 'import os\nimport sys'
    var_7 = 'os'
    var_8 = [var_7]
    var_9 = module_0.Config()



# Parsed testcases at query #9
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = module_1.Config()
    var_5 = 'test.py'
    var_6 = module_2.Path(var_5)
    var_7 = True
    var_8 = 'py'
    var_9 = 120



# Parsed testcases at query #10
#--------------------------


import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 'skipped_file.py'
    var_5 = module_1.Path(var_4)
    var_6 = 'import sys\nimport os\ninvalid syntax here\n'
    var_7 = 'error_file.py'
    var_8 = module_1.Path(var_7)
    var_9 = '# isort: skip_file\nimport sys\nimport os\n'
    var_10 = 'skip_comment_file.py'
    var_11 = module_1.Path(var_10)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'correct.py'
    var_1 = 'import os\nimport sys\n'
    assert var_1 is False
    var_2 = 'incorrect.py'
    var_3 = 'import sys\nimport os\n'
    var_4 = 'diff.py'
    var_5 = True
    var_6 = 'skipped.py'
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = 'syntax_error.py'
    var_10 = 'import sys\nimport os\nif\n'
    var_11 = 'non_existent.py'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os\n\ndef foo():\n    import sys'
    var_4 = 'import os\nimport sys'
    var_5 = module_0.Config()
    var_6 = ''
    var_7 = 'import os'
    var_8 = 'test.py'
    var_9 = module_1.Path(var_8)



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.Config()
    var_2 = 'import sys\nimport os\n'
    var_3 = module_0.Config()
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = module_0.Config()
    var_7 = module_1.check_file(var_4, var_5, var_6)
    var_8 = 'import sys\nimport os\n'
    var_9 = 'test.py'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = False
    var_13 = 'non_existent_file.py'
    var_14 = module_0.Config()
    var_15 = module_1.check_file(var_13, config=var_14)



# Parsed testcases at query #14
#--------------------------


import zipfile as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport sys\nimport os'
    var_2 = True
    var_3 = 'import os\ndef foo():\n    import sys'
    var_4 = 'import os'
    var_5 = 'test.py'
    var_6 = module_0.Path(var_5)
    var_7 = ''
    var_8 = 'import os'
    var_9 = module_1.Config()



# Parsed testcases at query #15
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = False
    var_2 = 'import sys\nimport os\n'
    var_3 = False
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = module_0.check_file(var_4, var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)
    var_10 = 'non_existent_file.py'
    var_11 = module_0.check_file(var_10)
    var_12 = 'import sys\nimport os\ninvalid syntax\n'
    var_13 = module_0.check_file(var_12)



# Parsed testcases at query #16
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = module_1.Config()
    var_4 = 'test.py'
    var_5 = module_2.Path(var_4)
    var_6 = True
    var_7 = 'py'
    var_8 = 120



# Parsed testcases at query #17
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import os\nimport sys\nimport os\n'
    var_2 = True
    var_3 = 'import os\n\ndef foo():\n    import sys\n'
    var_4 = True
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nimport json\n'
    var_1 = 'import json\nimport os\nimport sys\n'
    var_2 = 'from x import y\nimport os\nimport sys\nfrom a import b\n'
    var_3 = 'import os\nimport sys\nimport json\n'
    var_4 = True
    var_5 = 'import os\nimport sys\nimport json\n'
    var_6 = True



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import b\nimport a\n'
    assert var_0 == 'import a\nimport b\n'
    var_1 = 'import a\nimport b\n'
    var_2 = 'import b\nimport a\n'
    var_3 = True
    var_4 = 'import b\nimport a\n'
    var_5 = True
    var_6 = 'import b\nimport a\n'
    var_7 = True
    assert var_7 == 'import a, b\n'
    var_8 = module_0.Config()



# Parsed testcases at query #20
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = True
    var_4 = module_1.Config()
    var_5 = 'from os import path\nimport sys\n'
    var_6 = 'test.py'
    var_7 = module_2.Path(var_6)
    var_8 = 'py'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'from os import path\nimport sys\n'
    assert var_0 == 'from os import path\nimport sys\n'
    var_1 = 'import sys\nfrom os import path\n'
    assert var_1 == 'from os import path\nimport sys\n'
    var_2 = 'import sys\nfrom os import path\n'
    var_3 = 'import sys\nfrom os import path\n'
    var_4 = True
    var_5 = 'import sys\nfrom os import path\n'
    var_6 = True
    assert var_6 == 'import sys\nfrom os import path\n'



# Parsed testcases at query #22
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import sys\nimport os\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)
    var_10 = 'non_existent_file.py'
    var_11 = module_0.check_file(var_10)



# Parsed testcases at query #23
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    var_7 = True



# Parsed testcases at query #25
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)
    var_10 = 'import sys\nimport os\n'
    var_11 = 'test.py'
    var_12 = [var_11]
    var_13 = module_1.Config()
    var_14 = False
    var_15 = module_0.check_file(var_10, config=var_13, disregard_skip=var_14)
    var_16 = 'import sys\nimport os\n'
    var_17 = 'pyi'
    var_18 = module_0.check_file(var_11, extension=var_17)



# Parsed testcases at query #26
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path'
    var_2 = 'test1.py'
    var_3 = 'import os\nimport sys'
    var_4 = 'test2.py'
    var_5 = 'from pathlib import Path\nimport json'
    var_6 = 'test.py'
    var_7 = 'import os\nimport sys\nimport os'
    var_8 = True
    var_9 = list(var_5)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'test.py'
    var_12 = 'import os\ndef foo():\n    import sys'
    var_13 = True
    var_14 = list(var_5)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'non_existent_file.py'
    var_17 = [var_16]
    var_18 = module_0.find_imports_in_paths(var_17)
    var_19 = list(var_18)



# Parsed testcases at query #27
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 == 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    assert var_1 == 'import os\nimport sys\n'
    var_2 = 'import os\nimport sys\n'
    var_3 = 'import sys\nimport os\n'
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = 'import sys\nimport os\n'
    var_7 = 50
    var_8 = 'non_existent_file.py'
    var_9 = module_0.sort_file(var_8)
    var_10 = 'import sys\nimport os\ninvalid syntax here\n'
    var_11 = module_0.sort_file(var_10)
    assert var_11 is False
    assert var_11 is True
    var_12 = 'import sys\nimport os\n'
    var_13 = True
    var_14 = 'import sys\nimport os\n'
    var_15 = True
    var_16 = 'import sys\nimport os\n'
    var_17 = True
    var_18 = module_0.sort_file(var_16, ask_to_apply=var_17)
    assert var_18 is False
    var_19 = 'import sys\nimport os\n'
    var_20 = 50
    var_21 = module_1.Config()
    var_22 = module_0.sort_file(var_15, config=var_21)
    assert var_22 is True



# Parsed testcases at query #28
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom typing import List\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #29
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path\nimport json'
    var_3 = 'import os\nimport os\nimport sys'
    var_4 = [var_2]
    var_5 = True
    var_6 = module_0.find_imports_in_paths(var_4, unique=var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'import os\ndef foo():\n    import sys'
    var_10 = [var_2]
    var_11 = True
    var_12 = module_0.find_imports_in_paths(var_10, top_only=var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'non_existent_file.py'
    var_16 = [var_15]
    var_17 = module_0.find_imports_in_paths(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0



# Parsed testcases at query #30
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_imports.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = module_1.find_imports_in_file(var_1)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 'import os\nimport os\nfrom pathlib import Path\n'
    var_7 = True
    var_8 = module_1.find_imports_in_file(var_1, unique=var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'import os\nimport os.path\nfrom pathlib import Path\n'
    var_12 = len(var_9)
    assert var_12 == 2
    var_13 = 'import os\n\ndef foo():\n    import sys\n'
    var_14 = module_1.find_imports_in_file(var_1, top_only=var_7)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'non_existent_file.py'
    var_18 = module_1.find_imports_in_file(var_17)
    var_19 = list(var_18)



# Parsed testcases at query #31
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 'import a\nimport b'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = True
    var_7 = module_0.StringIO()
    var_8 = 'py'
    var_9 = 'test.py'
    var_10 = module_1.Path(var_9)
    var_11 = [var_9]
    var_12 = module_2.Config()
    var_13 = module_0.StringIO()
    var_14 = 'test.py'
    var_15 = module_1.Path(var_14)
    var_16 = [var_14]
    var_17 = module_2.Config()



# Parsed testcases at query #32
#--------------------------


import _io as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    assert var_0 == 'import a\nimport b\n'
    var_1 = 'import b\nimport a\n'
    var_2 = module_0.StringIO()
    var_3 = 'import b\nimport a\n'
    var_4 = module_0.StringIO()
    var_5 = True
    var_6 = 'import b\nimport a\n'
    var_7 = True
    var_8 = module_1.sort_file(var_6, ask_to_apply=var_7)
    assert var_8 is False
    assert var_8 is True
    var_9 = 'import b\nimport a\n'
    var_10 = 50
    var_11 = 'import a\nimport b\n'
    var_12 = 'non_existent_file.py'
    var_13 = module_1.sort_file(var_12)



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    var_7 = module_0.Config()
    var_8 = False



# Parsed testcases at query #34
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 is True
    assert var_0 == 'import os\nimport sys\n'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'import sys\nimport os\n'
    assert var_2 is True
    var_3 = 0
    assert var_3 is True
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = 0
    var_7 = 'import sys\nimport os\n'
    assert var_7 == 'import os, sys\n'
    var_8 = True



# Parsed testcases at query #36
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = len(var_1)
    var_4 = module_0.StringIO()
    var_5 = 'test.py'
    var_6 = module_1.Path(var_5)
    var_7 = [var_5]
    var_8 = module_2.Config()
    var_9 = True
    var_10 = 120
    var_11 = module_2.Config()
    var_12 = 'py'
    var_13 = ''
    var_14 = 'import os\n'



# Parsed testcases at query #37
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.StringIO()
    var_2 = 'import sys\nimport os\n'
    var_3 = module_0.StringIO()
    var_4 = module_0.StringIO()
    var_5 = True
    var_6 = module_0.StringIO()
    var_7 = module_1.Config()
    var_8 = 'import os\nimport sys\ninvalid syntax\n'
    var_9 = module_0.StringIO()
    var_10 = module_1.Config()
    var_11 = module_0.StringIO()
    var_12 = 'test.py'
    var_13 = module_2.Path(var_12)
    var_14 = [var_12]
    var_15 = module_1.Config()
    var_16 = module_0.StringIO()
    var_17 = module_2.Path(var_12)
    var_18 = [var_12]
    var_19 = module_1.Config()



# Parsed testcases at query #38
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom typing import List\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\nclass MyClass:\n    import sys\n'



# Parsed testcases at query #40
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import os\nimport os\n'
    var_2 = True
    var_3 = 'from os import path\nfrom os import sep\n'
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'import os\nimport sys\n'
    var_6 = module_0.Config()



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = 'import os\nimport sys\n'
    var_4 = True
    var_5 = 'import os\nimport sys\n'



# Parsed testcases at query #42
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import sys\nfrom os import path\nimport numpy as np\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import sys\n\ndef foo():\n    import os\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #43
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = len(var_1)
    var_4 = module_0.StringIO()
    var_5 = 'test.py'
    var_6 = module_1.Path(var_5)
    var_7 = [var_5]
    var_8 = module_2.Config()
    var_9 = True
    var_10 = 120
    var_11 = module_2.Config()
    var_12 = 'py'
    var_13 = ''
    var_14 = 'import os\n'
    var_15 = 'import os\nimport sys\nimport json\n'
    var_16 = 'from os import path\nfrom sys import argv\n'
    var_17 = 'import os\nfrom sys import argv\nimport json\n'



# Parsed testcases at query #44
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'non_existent_file.py'
    var_4 = module_0.find_imports_in_file(var_3)
    var_5 = list(var_4)
    var_6 = 'import os\nimport os\nimport sys\n'
    var_7 = True
    var_8 = list(var_4)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'import os\n\ndef foo():\n    import sys\n'
    var_11 = True
    var_12 = list(var_4)
    var_13 = len(var_12)
    assert var_13 == 1



# Parsed testcases at query #45
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    assert var_2 is True
    var_3 = module_0.StringIO()
    var_4 = module_1.Config()
    var_5 = 'test.py'
    var_6 = module_2.Path(var_5)
    var_7 = True
    var_8 = 'py'
    var_9 = module_1.Config()



# Parsed testcases at query #46
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #47
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nimport sys'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 0
    var_5 = 'import os.path\nimport os'
    var_6 = 'import os.path\nimport os.sys'
    var_7 = 'import os\n\ndef foo():\n    import sys'
    var_8 = 'import os\nimport sys'
    var_9 = 'force_single_line'
    var_10 = {var_9: var_2}



# Parsed testcases at query #49
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_2 = ''
    var_3 = 'x = 1\ny = 2\n'
    var_4 = 'import os\nx = \n'
    var_5 = '# isort: skip\nimport sys\nimport os\n'
    var_6 = 'import sys\nimport os\n'
    var_7 = module_0.StringIO()
    var_8 = True
    var_9 = module_1.Config()
    var_10 = 'from pathlib import (Path, PurePath)\n'
    var_11 = 'test.py'
    var_12 = module_2.Path(var_11)
    var_13 = 'py'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'import sys\nimport os\n'



# Parsed testcases at query #51
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)



# Parsed testcases at query #52
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import sys\nimport os\n'
    var_6 = 79
    var_7 = module_1.Config()
    var_8 = 'import sys\nimport os\n'
    var_9 = 'test.py'
    var_10 = [var_9]
    var_11 = module_1.Config()
    var_12 = False



# Parsed testcases at query #53
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = '# isort: skip_file\nimport sys\nimport os\n'
    var_3 = True
    var_4 = False
    var_5 = module_0.check_file(var_2, disregard_skip=var_4)
    var_6 = 'import sys\nimport os\ninvalid syntax here\n'
    var_7 = module_0.check_file(var_6)
    var_8 = 'import sys\nimport os\n'
    var_9 = 50
    var_10 = module_1.Config()
    var_11 = module_0.check_file(var_3, config=var_10)
    assert var_11 is True
    var_12 = 'import sys\nimport os\n'
    var_13 = 'pyx'
    var_14 = module_0.check_file(var_9, extension=var_13)
    assert var_14 is True
    var_15 = 'nonexistent_file.py'
    var_16 = module_0.check_file(var_15)



# Parsed testcases at query #54
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'non_existent_file.py'
    var_4 = module_0.find_imports_in_file(var_3)
    var_5 = list(var_4)
    var_6 = 'import os\nimport sys\nimport os\n'
    var_7 = True
    var_8 = list(var_4)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'import os\n\ndef foo():\n    import sys\n'
    var_11 = True
    var_12 = list(var_4)
    var_13 = len(var_12)
    assert var_13 == 1



# Parsed testcases at query #55
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_2 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = 'non_existent_file.py'
    var_8 = module_0.check_file(var_7)



# Parsed testcases at query #56
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nimport sys'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 'from os import path\nfrom os import path as path_module'
    var_5 = 'import os.path\nimport os'
    var_6 = 'import os.path\nimport os.sys'
    var_7 = 'import os\n\ndef func():\n    import sys'
    var_8 = ''
    var_9 = 'import os\nimport sys'
    var_10 = 'os'
    var_11 = [var_10]
    var_12 = module_0.Config()



# Parsed testcases at query #57
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.StringIO()
    var_2 = 0
    var_3 = 'import sys\n\nimport os\n'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = True
    var_7 = module_0.StringIO()
    var_8 = module_1.Config()
    var_9 = module_0.StringIO()
    var_10 = 'test.py'
    var_11 = module_2.Path(var_10)
    var_12 = 'py'
    var_13 = module_0.StringIO()
    var_14 = [var_10]
    var_15 = module_1.Config()
    var_16 = module_2.Path(var_10)
    var_17 = module_0.StringIO()
    var_18 = module_1.Config()
    var_19 = 'import os\nimport sys\ninvalid syntax\n'
    var_20 = module_0.StringIO()
    var_21 = module_1.Config()



# Parsed testcases at query #58
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nimport json\n'
    var_1 = 'import json\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nimport json\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nimport json\n'
    var_5 = True
    var_6 = 'import os\nimport sys\nimport json\n'
    var_7 = module_0.StringIO()
    var_8 = 0



# Parsed testcases at query #59
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 is True
    assert var_0 == 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    assert var_1 is True
    var_2 = True
    var_3 = 'line_length'
    var_4 = 50
    var_5 = {var_3: var_4}
    var_6 = 'non_existent_file.py'
    var_7 = module_0.sort_file(var_6)
    var_8 = 'import os\nimport sys\ninvalid syntax here\n'



# Parsed testcases at query #60
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'non_existent_file.py'
    var_6 = module_0.check_file(var_5)
    var_7 = 'import os\nimport\n'
    var_8 = module_0.check_file(var_7)
    var_9 = '# isort: skip_file\nimport sys\nimport os\n'
    var_10 = False
    var_11 = module_0.check_file(var_9, disregard_skip=var_10)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_5 = True



# Parsed testcases at query #2
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)
    var_10 = 79
    var_11 = module_1.Config()
    var_12 = 'import os\nimport sys\n'
    var_13 = 'non_existent_file.py'
    var_14 = module_0.check_file(var_13)



# Parsed testcases at query #3
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = module_0.StringIO()
    var_4 = module_1.Config()
    var_5 = 'test.py'
    var_6 = module_2.Path(var_5)
    var_7 = True
    var_8 = 'py'
    var_9 = 120



# Parsed testcases at query #4
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'def foo():\n    pass\n'
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path\n'
    var_7 = True
    var_8 = list(var_5)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 'import os\n\ndef foo():\n    import sys\n'
    var_11 = True
    var_12 = list(var_5)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'non_existent_file.py'
    var_15 = module_0.find_imports_in_file(var_14)
    var_16 = list(var_15)



# Parsed testcases at query #5
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 79
    var_9 = module_1.Config()
    var_10 = 'import os\nimport sys\n'



# Parsed testcases at query #6
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test.py'
    var_8 = module_2.Path(var_7)
    var_9 = [var_4]
    var_10 = module_1.Config()
    var_11 = module_2.Path(var_4)
    var_12 = True
    var_13 = '.py'
    var_14 = 120
    var_15 = module_1.Config()



# Parsed testcases at query #7
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b'
    var_3 = module_0.StringIO()
    var_4 = module_0.StringIO()
    var_5 = True
    var_6 = module_1.Config()
    var_7 = module_0.StringIO()
    var_8 = 'test.py'
    var_9 = module_2.Path(var_8)
    var_10 = module_0.StringIO()
    var_11 = module_0.StringIO()
    var_12 = module_0.StringIO()
    var_13 = False



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 is True
    assert var_0 == 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    assert var_1 is True
    var_2 = 0
    assert var_2 is True
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    assert var_6 is True
    var_7 = 0



# Parsed testcases at query #9
#--------------------------


import isort.api as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)
    var_10 = 'import sys\nimport os\n'
    var_11 = 'test.py'
    var_12 = [var_11]
    var_13 = module_1.Config()
    var_14 = False
    var_15 = module_0.check_file(var_10, config=var_13, disregard_skip=var_14)
    var_16 = 'import sys\nimport os\n'
    var_17 = 50
    var_18 = module_1.Config()
    var_19 = module_0.check_file(var_12, config=var_18)
    assert var_19 is False
    var_20 = 'import sys\nimport os\n'
    var_21 = 'pyx'
    var_22 = module_0.check_file(var_17, extension=var_21)
    assert var_22 is False
    var_23 = 'import sys\nimport os\n'
    var_24 = 'custom/path.py'
    var_25 = module_2.Path(var_24)
    var_26 = module_0.check_file(var_17, file_path=var_25)
    assert var_26 is False
    var_27 = 'import sys\nimport os\n'
    var_28 = 50
    var_29 = module_0.check_file(var_17)
    assert var_29 is False



# Parsed testcases at query #10
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #11
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_5 = True
    var_6 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_7 = module_0.StringIO()
    var_8 = 0



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = 'from x import y\nimport z\n'
    var_5 = True
    var_6 = 'import sys\nimport os\n'
    var_7 = False
    var_8 = 'test*.py'
    var_9 = [var_8]



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 'from os import path\nfrom os import path as os_path'
    var_5 = 'import os\nfrom os import path'
    var_6 = 'import os.path\nimport os.environ'
    var_7 = 'import os\n\ndef foo():\n    import sys'
    var_8 = ''
    var_9 = 'def foo():\n    pass'



# Parsed testcases at query #14
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax here\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)



# Parsed testcases at query #15
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = 'import os\nimport sys\ninvalid syntax\n'
    var_6 = module_0.check_file(var_5)
    var_7 = '# isort: skip_file\nimport sys\nimport os\n'
    var_8 = False
    var_9 = module_0.check_file(var_7, disregard_skip=var_8)



# Parsed testcases at query #16
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path'
    var_2 = 'file2.py'
    var_3 = 'import json\nfrom typing import List'
    var_4 = False
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = 'pathlib'
    var_8 = 'json'
    var_9 = 'typing'
    var_10 = 'file3.py'
    var_11 = 'import os\n\ndef foo():\n    import sys'
    var_12 = True
    var_13 = 'non_existent_path'
    var_14 = [var_13]
    var_15 = module_0.find_imports_in_paths(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = 'empty'
    var_19 = len(var_16)
    assert var_19 == 0



# Parsed testcases at query #17
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #18
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'import sys\nimport os\ninvalid syntax here\n'
    var_9 = module_0.check_file(var_8)
    var_10 = 'import sys\nimport os\ninvalid syntax here\n'



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    var_7 = module_0.Config()
    var_8 = False



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import os\nimport sys\n'
    var_7 = True



# Parsed testcases at query #21
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = 'test.py'
    var_4 = module_1.Path(var_3)
    var_5 = module_2.Config()
    var_6 = module_1.Path(var_3)
    var_7 = [var_3]
    var_8 = module_2.Config()
    var_9 = True
    var_10 = 'py'
    var_11 = 120



# Parsed testcases at query #22
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #23
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path\nimport numpy as np\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = 'import os\n\ndef foo():\n    import sys\n'
    var_4 = True
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #24
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 79
    var_9 = module_1.Config()
    var_10 = 'import os\nimport sys\n'
    var_11 = 'non_existent_file.py'
    var_12 = module_0.check_file(var_11)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom os import path\nfrom os import path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom os import path\nfrom os import path'
    var_4 = 'from os import path\nfrom os import path\nfrom os import mkdir'
    var_5 = 'import os\n\ndef foo():\n    import sys'
    var_6 = ''
    var_7 = 'def foo():\n    pass'



# Parsed testcases at query #26
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom typing import List\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #27
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = 'test.py'
    var_4 = module_1.Path(var_3)
    var_5 = 'py'
    var_6 = module_1.Path(var_3)
    var_7 = [var_3]
    var_8 = module_2.Config()
    var_9 = True



# Parsed testcases at query #28
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'print("Hello, World!")\n'
    var_2 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = True
    var_6 = 'non_existent_file.py'
    var_7 = module_0.find_imports_in_file(var_6)
    var_8 = list(var_7)



# Parsed testcases at query #29
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 0
    var_3 = 'import a\nimport b'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = 'test.py'
    var_7 = module_1.Path(var_6)
    var_8 = 'py'
    var_9 = module_2.Config()
    var_10 = module_0.StringIO()
    var_11 = module_0.StringIO()
    var_12 = True
    var_13 = module_0.StringIO()
    var_14 = module_0.StringIO()
    var_15 = False



# Parsed testcases at query #30
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.sort_file(var_1)
    assert var_3 is True
    var_4 = var_1.read_text()
    assert var_4 == 'import os\nimport sys\n'



# Parsed testcases at query #31
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 'test.py'
    var_5 = module_1.Path(var_4)
    var_6 = module_1.Path(var_4)
    var_7 = [var_4]
    var_8 = module_2.Config()
    var_9 = True
    var_10 = 'py'
    var_11 = 120



# Parsed testcases at query #32
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 'import a\nimport b\n'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = True
    var_7 = 'test.py'
    var_8 = module_1.Path(var_7)
    var_9 = module_2.Config()
    var_10 = module_0.StringIO()
    var_11 = module_0.StringIO()
    var_12 = module_2.Config()
    var_13 = module_0.StringIO()
    var_14 = module_2.Config()
    var_15 = 'import b\nimport a\ninvalid syntax\n'
    var_16 = module_0.StringIO()
    var_17 = 'py'



# Parsed testcases at query #33
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path'
    var_2 = 'file2.py'
    var_3 = 'import json\nfrom typing import List'
    var_4 = 'os'
    var_5 = 'sys'
    var_6 = 'pathlib'
    var_7 = 'json'
    var_8 = 'typing'
    var_9 = 'file1.py'
    var_10 = 'import os\nimport sys\nimport os'
    var_11 = True
    var_12 = list(var_3)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'file1.py'
    var_15 = 'import os\n\ndef foo():\n    import sys'
    var_16 = True
    var_17 = list(var_3)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = list(var_14)
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = 'non_existent_path'
    var_22 = [var_21]
    var_23 = module_0.find_imports_in_paths(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0



# Parsed testcases at query #34
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 'test.py'
    var_5 = module_1.Path(var_4)
    var_6 = [var_4]
    var_7 = module_2.Config()
    var_8 = True
    var_9 = 120
    var_10 = module_2.Config()



# Parsed testcases at query #35
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 'test.py'
    var_5 = module_1.Path(var_4)
    var_6 = [var_4]
    var_7 = module_2.Config()
    var_8 = True
    var_9 = '.py'
    var_10 = 120



# Parsed testcases at query #36
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_1 = 'import json\nimport os\nimport sys\n\nfrom pathlib import Path\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_5 = True
    var_6 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_7 = module_0.StringIO()
    var_8 = 0



# Parsed testcases at query #37
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom os import path'
    var_4 = 'from os import path\nfrom os import path\nfrom pathlib import Path'
    var_5 = 'import os\ndef foo():\n    import sys'
    var_6 = ''
    var_7 = 'import os\nimport sys'
    var_8 = module_0.Config()



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 'from os import path\nfrom os import path\nfrom os import listdir'
    var_5 = 'import os\nimport os.path\nfrom os import path'
    var_6 = 'import os.path\nimport os.listdir\nimport sys.path'
    var_7 = 'import os\n\ndef function():\n    import sys'
    var_8 = ''
    var_9 = 'def function():\n    pass'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_2 = ''
    var_3 = 'x = 1\ny = 2\n'
    var_4 = 'import sys\nx = 1\nimport os\n'
    var_5 = 'from os import path\nfrom sys import argv\n'
    var_6 = 'from sys import argv\nfrom os import path\n'
    var_7 = 'import numpy as np\nimport pandas as pd\n'
    var_8 = 'import pandas as pd\nimport numpy as np\n'
    var_9 = 'from . import module\nfrom .. import module\n'
    var_10 = 'from .. import module\nfrom . import module\n'
    var_11 = 'from os import *\nfrom sys import *\n'
    var_12 = 'from sys import *\nfrom os import *\n'
    var_13 = '# This is a comment\nimport os\nimport sys\n'
    var_14 = '# This is a comment\nimport sys\nimport os\n'
    var_15 = 'from pathlib import (\n    Path,\n    PurePath,\n)\n'
    var_16 = 'from pathlib import (\n    PurePath,\n    Path,\n)\n'
    var_17 = 'from typing import Any, Dict, List\n'
    var_18 = 'from typing import Dict, Any, List\n'
    var_19 = 'if True:\n    import os\n    import sys\n'
    var_20 = 'if True:\n    import sys\n    import os\n'



# Parsed testcases at query #40
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import os\n'
    var_2 = 'import sys\n'
    var_3 = 'import os\nimport os\n'
    var_4 = True
    var_5 = 'import os\ndef foo():\n    import sys\n'
    var_6 = True
    var_7 = '/non/existent/path'
    var_8 = [var_7]
    var_9 = module_0.find_imports_in_paths(var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'import os\n'
    var_13 = module_0.find_imports_in_paths(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1



# Parsed testcases at query #41
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom pathlib import Path\nfrom pathlib import Path'
    var_4 = 'from os import path\nfrom os import path\nfrom pathlib import Path\nfrom pathlib import Path'
    var_5 = 'import os\n\ndef foo():\n    import sys\n    pass'
    var_6 = ''
    var_7 = 'import os\nimport sys'
    var_8 = 'os'
    var_9 = [var_8]
    var_10 = module_0.Config()



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 'from os import path\nfrom os import path\nfrom sys import path'
    var_5 = 'import os\nimport os.path\nfrom os import path'
    var_6 = 'import os.path\nimport os.sys\nfrom os import path'
    var_7 = 'import os\n\ndef foo():\n    import sys\n    pass'
    var_8 = ''
    var_9 = 'def foo():\n    pass'



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = True
    var_2 = True
    var_3 = 0
    var_4 = 'import os\nimport sys\n'



# Parsed testcases at query #44
#--------------------------


import _io as module_0
import zipfile as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = module_0.StringIO()
    var_4 = True
    var_5 = 'test.py'
    var_6 = module_1.Path(var_5)
    var_7 = [var_5]
    var_8 = module_2.Config()
    var_9 = module_1.Path(var_5)
    var_10 = 'py'
    var_11 = 120



# Parsed testcases at query #45
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import os\nfrom pathlib import Path\n'
    var_2 = 'import sys\nfrom collections import defaultdict\n'
    var_3 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path\n'
    var_4 = [var_2]
    var_5 = True
    var_6 = module_0.find_imports_in_paths(var_4, unique=var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'import os\n\ndef foo():\n    import sys\n'
    var_10 = [var_2]
    var_11 = True
    var_12 = module_0.find_imports_in_paths(var_10, top_only=var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'import os\nimport os.path\nfrom pathlib import Path\n'
    var_16 = [var_2]
    var_17 = module_0.find_imports_in_paths(var_16, unique=var_11)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = '/non/existent/path.py'
    var_21 = [var_20]
    var_22 = module_0.find_imports_in_paths(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0



# Parsed testcases at query #46
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = 'import os\nimport os'
    var_2 = True
    var_3 = 'import os\ndef foo():\n    import sys'
    var_4 = 'from os import path\nfrom sys import argv'
    var_5 = 'import os as operating_system\nimport os as os_alias'
    var_6 = 0
    var_7 = 'import os\nimport os.path'
    var_8 = 'from os import path\nfrom os import path as os_path'
    var_9 = 'import os.path\nimport os.sys'
    var_10 = ''
    var_11 = 'x = 1\ny = 2'
    var_12 = 'import os\nimport sys'
    var_13 = module_0.Config()



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = ''
    var_3 = 'x = 1\n'
    var_4 = 'x = 1\nimport sys\nimport os\n'
    var_5 = 'from os import path\nfrom sys import argv\n'
    var_6 = 'from sys import argv\nfrom os import path\n'
    var_7 = 'from . import module\nfrom .. import module\n'
    var_8 = 'from .. import module\nfrom . import module\n'
    var_9 = '# This is a comment\nimport os\nimport sys\n'
    var_10 = '# This is a comment\nimport sys\nimport os\n'
    var_11 = 'from os import path, environ\nfrom sys import argv, exit\n'
    var_12 = 'from sys import argv, exit\nfrom os import path, environ\n'
    var_13 = 'import os as operating_system\nimport sys as system\n'
    var_14 = 'import sys as system\nimport os as operating_system\n'



# Parsed testcases at query #48
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #49
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'non_existent_file.py'
    var_6 = module_0.find_imports_in_file(var_5)
    var_7 = list(var_6)



# Parsed testcases at query #50
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = module_0.StringIO()
    var_3 = module_1.Config()
    var_4 = 'test.py'
    var_5 = module_2.Path(var_4)
    var_6 = True



# Parsed testcases at query #51
#--------------------------


import _io as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 == 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    assert var_1 == 'import os\nimport sys\n'
    var_2 = 50
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 'import sys\nimport os\n'
    var_6 = True
    assert var_6 == 'import os\nimport sys\n'
    var_7 = 'import sys\nimport os\n'
    var_8 = module_0.StringIO()
    var_9 = 'import sys\nimport os\ninvalid syntax here\n'
    var_10 = module_1.sort_file(var_9)
    var_11 = 'import os\nimport sys\n'
    var_12 = 'import sys\nimport os\n'
    var_13 = True
    var_14 = module_1.sort_file(var_12, ask_to_apply=var_13)
    assert var_14 is False



# Parsed testcases at query #52
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\nfrom pathlib import Path'
    var_3 = module_1.find_imports_in_file(var_1)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 'import os\nimport os\nimport sys'
    var_7 = True
    var_8 = module_1.find_imports_in_file(var_1, unique=var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'import os\ndef foo():\n    import sys'
    var_12 = module_1.find_imports_in_file(var_1, top_only=var_7)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'non_existent_file.py'
    var_16 = module_1.find_imports_in_file(var_15)
    var_17 = list(var_16)



# Parsed testcases at query #53
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.check_file(var_2, var_3)
    var_5 = '# isort: skip_file\nimport sys\nimport os\n'
    var_6 = False
    var_7 = module_0.check_file(var_5, disregard_skip=var_6)
    var_8 = 'non_existent_file.py'
    var_9 = module_0.check_file(var_8)



# Parsed testcases at query #54
#--------------------------


import isort.settings as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'from collections import defaultdict\nfrom typing import List\n'
    var_2 = 'import os\nimport os\n'
    var_3 = True
    var_4 = 'import os\n\ndef foo():\n    import sys\n'
    var_5 = 'import os\nimport sys\n'
    var_6 = module_0.Config()
    var_7 = ''
    var_8 = 'import os\n'
    var_9 = 'test.py'
    var_10 = module_1.Path(var_9)



