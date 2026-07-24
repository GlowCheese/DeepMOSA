####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 is True
    var_1 = 'import os\nimport sys\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = 0
    var_7 = 'import sys\nimport os\n'
    assert var_7 is True
    var_8 = 0



# Parsed testcases at query #2
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.find_imports_in_code(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'from pathlib import Path\nfrom typing import List'
    var_5 = module_0.find_imports_in_code(var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'import os\nimport os'
    var_9 = True
    var_10 = module_0.find_imports_in_code(var_8, unique=var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'import os\ndef foo():\n    import sys'
    var_14 = module_0.find_imports_in_code(var_13, top_only=var_9)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'import os'
    var_18 = module_1.Config()
    var_19 = module_0.find_imports_in_code(var_17, var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = ''
    var_23 = module_0.find_imports_in_code(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = 'import os\nfrom pathlib import Path\nimport sys'
    var_27 = module_0.find_imports_in_code(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 3



# Parsed testcases at query #3
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\nfrom pathlib import Path\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = module_0.StringIO()
    var_4 = module_1.Config()
    var_5 = 'import os\nimport sys\n'
    var_6 = 'test.py'
    var_7 = module_2.Path(var_6)
    var_8 = True



# Parsed testcases at query #4
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = [var_1]
    var_4 = module_1.find_imports_in_paths(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 'test_file2.py'
    var_8 = module_0.Path(var_7)
    var_9 = 'import json\nfrom typing import List\n'
    var_10 = [var_1, var_8]
    var_11 = module_1.find_imports_in_paths(var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = 'test_file3.py'
    var_15 = module_0.Path(var_14)
    var_16 = 'import os\nimport sys\nimport os\n'
    var_17 = [var_15]
    var_18 = True
    var_19 = module_1.find_imports_in_paths(var_17, unique=var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'test_file4.py'
    var_23 = module_0.Path(var_22)
    var_24 = 'import os\n\ndef func():\n    import sys\n'
    var_25 = [var_23]
    var_26 = module_1.find_imports_in_paths(var_25, top_only=var_18)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = 'non_existent_file.py'
    var_30 = module_0.Path(var_29)
    var_31 = [var_30]
    var_32 = module_1.find_imports_in_paths(var_31)
    var_33 = list(var_32)



# Parsed testcases at query #5
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = True
    var_3 = module_0.StringIO()
    var_4 = 0
    var_5 = module_1.Config()
    var_6 = 'test.py'
    var_7 = module_2.Path(var_6)
    var_8 = [var_6]
    var_9 = module_1.Config()
    var_10 = module_2.Path(var_6)
    var_11 = True
    var_12 = 'py'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Path(var_0)
    var_2 = True
    var_3 = 'test_file.py'
    var_4 = var_1 / var_3
    var_5 = 'import os\nimport sys\nfrom pathlib import Path'
    var_6 = [var_1]
    var_7 = module_1.find_imports_in_paths(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 3



# Parsed testcases at query #8
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'def foo():\n    pass\n'
    var_2 = 'nonexistent_file.py'
    var_3 = module_0.find_imports_in_file(var_2)
    var_4 = list(var_3)
    var_5 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path\n'
    var_6 = True
    var_7 = 'import os as operating_system\nimport os as os_alias\nfrom pathlib import Path as P\nfrom pathlib import Path as PathAlias\n'
    var_8 = 0
    var_9 = 1
    var_10 = 'import os\n\ndef foo():\n    import sys\n    pass\n'
    var_11 = True



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    assert var_0 is True
    assert var_0 == 'import sys\n\nimport os\n'
    var_1 = 'import os\nimport sys\n'
    assert var_1 is True
    var_2 = 0
    assert var_2 is True
    var_3 = True
    var_4 = 0
    var_5 = 50



# Parsed testcases at query #10
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import os\nimport sys\n'
    var_3 = 'import sys\nimport os\n'
    var_4 = module_0.StringIO()
    var_5 = 'import sys\nimport os\n'
    var_6 = module_0.StringIO()
    var_7 = True
    var_8 = 'import sys\nimport os\n'
    var_9 = 120



# Parsed testcases at query #11
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 'import a\nimport b'
    var_4 = module_0.StringIO()
    var_5 = module_0.StringIO()
    var_6 = True
    var_7 = module_1.Config()
    var_8 = module_0.StringIO()
    var_9 = module_0.StringIO()
    var_10 = 'test.py'
    var_11 = module_2.Path(var_10)
    var_12 = module_0.StringIO()
    var_13 = module_2.Path(var_10)
    var_14 = module_0.StringIO()
    var_15 = module_1.Config()



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'from pathlib import Path\nimport os\nimport sys\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_5 = True



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = 'import sys\nimport os\n'
    var_5 = True
    var_6 = 'import os\nimport sys\n'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'from pathlib import Path\n\nimport json\nimport os\nimport sys\n'

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_2 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_2 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_2 = 'isort.ask_whether_to_apply_changes_to_file'
    var_3 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_2 = 'isort.ask_whether_to_apply_changes_to_file'
    var_3 = False
    var_4 = True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\nif\n'

import _io as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_2 = module_0.StringIO()
    var_3 = 0



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import os\nimport sys\nimport os\n'
    var_2 = True
    var_3 = 'import os as operating_system\nimport sys as system\nimport os as operating_system\n'
    var_4 = 'import os\nfrom os import path\nimport os\n'
    var_5 = 'from os import path\nfrom os import path\nfrom os import path\n'
    var_6 = 'import os\n\ndef foo():\n    import sys\n'
    var_7 = ''
    var_8 = 'import os\nimport sys\n'
    var_9 = module_0.Config()



# Parsed testcases at query #17
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
    var_10 = 'non_existent_file.py'
    var_11 = module_0.check_file(var_10)



# Parsed testcases at query #19
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
    var_12 = 'import b\nimport a\ninvalid syntax\n'
    var_13 = module_1.sort_file(var_12)
    assert var_13 is False



# Parsed testcases at query #20
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



# Parsed testcases at query #21
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
    assert var_6 is True
    var_7 = 79
    var_8 = module_1.Config()
    var_9 = 'import os\nimport sys\n'



# Parsed testcases at query #22
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
    var_6 = 'py'
    var_7 = [var_4]
    var_8 = module_2.Config()
    var_9 = module_1.Path(var_4)
    var_10 = True
    var_11 = 120



# Parsed testcases at query #23
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 0
    var_4 = 'import a\nimport b'
    var_5 = module_0.StringIO()
    var_6 = module_0.StringIO()
    var_7 = 'py'
    var_8 = True
    var_9 = '---'
    var_10 = module_0.StringIO()
    var_11 = True
    var_12 = module_1.Config()
    var_13 = 'import b\nimport a\ninvalid syntax'
    var_14 = module_0.StringIO()
    var_15 = module_1.Config()
    var_16 = 'py'
    var_17 = module_0.StringIO()
    var_18 = 'test.py'
    var_19 = module_2.Path(var_18)
    var_20 = [var_18]
    var_21 = module_1.Config()



# Parsed testcases at query #24
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
    var_9 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_10 = 50



# Parsed testcases at query #25
#--------------------------


import zipfile as module_0
import isort.api as module_1
import _io as module_2

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_3 = module_1.sort_file(var_1)
    assert var_3 is True
    var_4 = var_1.read_text()
    assert var_4 == 'from pathlib import Path\nimport json\nimport os\nimport sys\n'
    var_5 = module_0.Path(var_0)
    var_6 = 'from pathlib import Path\nimport json\nimport os\nimport sys\n'
    var_7 = module_1.sort_file(var_5)
    assert var_7 is False
    var_8 = module_0.Path(var_0)
    var_9 = module_2.StringIO()
    var_10 = module_1.sort_file(var_8, show_diff=var_9)
    assert var_10 is True
    var_11 = module_0.Path(var_0)
    var_12 = module_2.StringIO()
    var_13 = True
    var_14 = module_1.sort_file(var_11, write_to_stdout=var_13, output=var_12)
    assert var_14 is True
    var_15 = module_0.Path(var_0)
    var_16 = True
    var_17 = module_1.sort_file(var_15, ask_to_apply=var_16)
    assert var_17 is False
    var_18 = var_15.read_text()
    assert var_18 == 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_19 = module_0.Path(var_16)
    var_20 = True
    var_21 = module_1.sort_file(var_19, ask_to_apply=var_20)
    assert var_21 is True
    var_22 = var_19.read_text()
    assert var_22 == 'from pathlib import Path\nimport json\nimport os\nimport sys\n'



# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 'import a\nimport b\n'
    var_4 = module_0.StringIO()
    var_5 = module_1.Config()
    var_6 = module_0.StringIO()
    var_7 = 'test.py'
    var_8 = module_2.Path(var_7)
    var_9 = module_0.StringIO()
    var_10 = module_0.StringIO()
    var_11 = True
    var_12 = module_0.StringIO()
    var_13 = module_0.StringIO()



# Parsed testcases at query #28
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = module_0.StringIO()
    var_5 = 120
    var_6 = module_1.Config()
    var_7 = module_0.StringIO()
    var_8 = 'test.py'
    var_9 = module_2.Path(var_8)
    var_10 = 'py'
    var_11 = module_0.StringIO()
    var_12 = True
    var_13 = module_0.StringIO()
    var_14 = module_2.Path(var_8)
    var_15 = [var_8]
    var_16 = module_1.Config()
    var_17 = module_0.StringIO()
    var_18 = module_1.Config()
    var_19 = 'import b\nimport a\ninvalid syntax\n'
    var_20 = module_0.StringIO()
    var_21 = module_1.Config()
    var_22 = module_0.StringIO()
    var_23 = module_1.Config()
    var_24 = 'test.pyx'
    var_25 = module_2.Path(var_24)
    var_26 = 'pyx'



# Parsed testcases at query #29
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
    var_8 = ''
    var_9 = 'import os\n'
    var_10 = 'import os\nimport sys\nimport json\n'
    var_11 = 'import os\nfrom sys import argv\nimport json\n'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path'
    var_2 = 'test2.py'
    var_3 = 'import json\nfrom typing import List'
    var_4 = True
    var_5 = 'test3.py'
    var_6 = 'import os\ndef foo():\n    import sys'
    var_7 = 'empty'
    var_8 = 'nonexistent'



# Parsed testcases at query #31
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 'import a\nimport b'
    var_4 = module_0.StringIO()
    var_5 = module_1.Config()
    var_6 = module_0.StringIO()
    var_7 = 'test.py'
    var_8 = module_2.Path(var_7)
    var_9 = module_0.StringIO()
    var_10 = module_0.StringIO()
    var_11 = True
    var_12 = module_0.StringIO()
    var_13 = module_0.StringIO()
    var_14 = module_1.Config()



# Parsed testcases at query #32
#--------------------------


import _io as module_0
import isort.settings as module_1
import zipfile as module_2

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'py'
    var_3 = 0
    var_4 = 'import a\nimport b'
    var_5 = module_0.StringIO()
    var_6 = module_0.StringIO()
    var_7 = True
    var_8 = module_1.Config()
    var_9 = module_0.StringIO()
    var_10 = 'py'
    var_11 = 'test.py'
    var_12 = module_2.Path(var_11)
    var_13 = [var_11]
    var_14 = module_1.Config()
    var_15 = 'invalid syntax'
    var_16 = module_0.StringIO()
    var_17 = 'py'
    var_18 = True
    var_19 = module_1.Config()
    var_20 = module_0.StringIO()
    var_21 = module_0.StringIO()



# Parsed testcases at query #33
#--------------------------


import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b\n'
    var_3 = module_0.StringIO()
    var_4 = module_0.StringIO()
    var_5 = 'test.py'
    var_6 = module_1.Path(var_5)
    var_7 = 'py'
    var_8 = module_0.StringIO()
    var_9 = module_0.StringIO()
    var_10 = module_0.StringIO()
    var_11 = 50
    var_12 = 'import b\nimport a\ninvalid syntax\n'
    var_13 = module_0.StringIO()
    var_14 = module_1.Path(var_5)
    var_15 = True



# Parsed testcases at query #34
#--------------------------


import isort.api as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.find_imports_in_paths(var_0)
    var_2 = list(var_1)
    var_3 = 'non_existent_path.py'
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
    var_17 = 'import datetime\nfrom collections import defaultdict'
    var_18 = [var_13, var_15]
    var_19 = module_0.find_imports_in_paths(var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 'import os\nimport sys\nimport os'
    var_23 = [var_14]
    var_24 = True
    var_25 = module_0.find_imports_in_paths(var_23, unique=var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = 'import os\ndef foo():\n    import sys'
    var_29 = [var_14]
    var_30 = True
    var_31 = module_0.find_imports_in_paths(var_29, top_only=var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = 'my_package'
    var_35 = [var_34]
    var_36 = module_1.Config()
    var_37 = 'import my_package\nimport os'
    var_38 = [var_14]
    var_39 = module_0.find_imports_in_paths(var_38, var_36)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 2



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'from x import y\nimport z\n'
    var_7 = True



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_2 = True
    var_3 = 'import os\n\ndef foo():\n    import sys\n'
    var_4 = 'non_existent.py'
    var_5 = list(var_0)



# Parsed testcases at query #3
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
    var_8 = True
    var_9 = True



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_1 = 'import json\nimport os\nimport sys\n\nfrom pathlib import Path\n'
    var_2 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_3 = True
    var_4 = 'import os\nimport sys\nfrom pathlib import Path\nimport json\n'
    var_5 = True



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nimport sys'
    var_2 = True
    var_3 = 'import os as operating_system\nimport os as os_module'
    var_4 = 0
    var_5 = 'from os import path\nfrom os import path as os_path'
    var_6 = 'import os\nimport os.path\nfrom os import path'
    var_7 = 'import os.path\nimport os.sys\nfrom os import path'
    var_8 = 'import os\n\ndef foo():\n    import sys'
    var_9 = ''
    var_10 = 'import os\nimport sys'
    var_11 = module_0.Config()



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------


import _io as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 'import sys\nimport os\n'
    var_3 = True
    var_4 = module_0.StringIO()
    var_5 = 0
    var_6 = 50



# Parsed testcases at query #8
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
    assert var_6 is False
    var_7 = 'import sys\nimport os\ninvalid syntax here\n'
    var_8 = module_0.check_file(var_7)
    var_9 = 'import sys\nimport os\ninvalid syntax here\n'



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


import zipfile as module_0
import isort.api as module_1
import _io as module_2
import isort.settings as module_3

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Path(var_0)
    var_2 = 'import b\nimport a\n'
    var_3 = module_1.sort_file(var_1)
    assert var_3 is True
    var_4 = var_1.read_text()
    assert var_4 == 'import a\nimport b\n'
    var_5 = 'import a\nimport b\n'
    var_6 = module_1.sort_file(var_1)
    assert var_6 is False
    var_7 = var_1.read_text()
    assert var_7 == 'import a\nimport b\n'
    var_8 = module_2.StringIO()
    var_9 = module_1.sort_file(var_1, show_diff=var_8)
    assert var_9 is True
    var_10 = module_2.StringIO()
    var_11 = True
    var_12 = module_1.sort_file(var_1, write_to_stdout=var_11)
    assert var_12 is True
    var_13 = module_2.StringIO()
    var_14 = module_1.sort_file(var_1, output=var_13)
    assert var_14 is True
    var_15 = 0
    var_16 = True
    var_17 = module_1.sort_file(var_1, ask_to_apply=var_16)
    assert var_17 is False
    var_18 = var_1.read_text()
    assert var_18 == 'import b\nimport a\n'
    var_19 = 'from x import b\nfrom x import a\n'
    var_20 = True
    var_21 = module_1.sort_file(var_1)
    assert var_21 is True
    var_22 = var_1.read_text()
    var_23 = [var_16]
    var_24 = module_3.Config()
    var_25 = False
    var_26 = module_1.sort_file(var_1, config=var_24, disregard_skip=var_25)
    assert var_26 is False



# Parsed testcases at query #11
#--------------------------


import isort.api as module_0
import isort.settings as module_1

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
    var_9 = module_0.check_file(var_7, var_8, disregard_skip=var_8)
    var_10 = 'import sys\nimport os\ninvalid syntax here\n'
    var_11 = False
    var_12 = module_0.check_file(var_10, var_11)
    var_13 = 'import sys\nimport os\ninvalid syntax here\n'
    var_14 = False
    var_15 = 79
    var_16 = module_1.Config()
    var_17 = 'import os\nimport sys\n'
    var_18 = False



# Parsed testcases at query #12
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'test1.py'
    var_3 = 'import os\nimport sys\n'
    var_4 = 'test2.py'
    var_5 = 'import json\nimport re\n'
    var_6 = 'test1.py'
    var_7 = 'import os\nimport sys\n'
    var_8 = 'test2.py'
    var_9 = 'import os\nimport re\n'
    var_10 = True
    var_11 = 'test1.py'
    var_12 = 'import os\nimport sys\n\ndef foo():\n    import json\n'
    var_13 = 'test2.py'
    var_14 = 'import os\nimport re\n\ndef bar():\n    import sys\n'
    var_15 = True
    var_16 = []
    var_17 = module_0.find_imports_in_paths(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = 'non_existent_path'
    var_21 = [var_20]
    var_22 = module_0.find_imports_in_paths(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 0



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    var_7 = True
    var_8 = 'import os\nimport sys\n'



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
    var_8 = 79
    var_9 = module_1.Config()
    var_10 = 'import os\nimport sys\n'
    var_11 = 'non_existent_file.py'
    var_12 = module_0.check_file(var_11)



# Parsed testcases at query #15
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Path(var_0)
    var_2 = True
    var_3 = 'file1.py'
    var_4 = var_1 / var_3
    var_5 = 'import os\nimport sys\nfrom pathlib import Path'
    var_6 = 'file2.py'
    var_7 = var_1 / var_6
    var_8 = 'import sys\nimport os\nfrom pathlib import Path'
    var_9 = [var_1]
    var_10 = module_1.find_imports_in_paths(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 6
    var_13 = [var_1]
    var_14 = module_1.find_imports_in_paths(var_13, unique=var_2)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = 'file3.py'
    var_18 = var_1 / var_17
    var_19 = 'import os\n\ndef func():\n    import sys'
    var_20 = [var_1]
    var_21 = module_1.find_imports_in_paths(var_20, top_only=var_2)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 7
    var_24 = 'non_existent'
    var_25 = module_0.Path(var_24)
    var_26 = [var_25]
    var_27 = module_1.find_imports_in_paths(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 0



# Parsed testcases at query #16
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Path(var_0)
    var_2 = True
    var_3 = 'file1.py'
    var_4 = var_1 / var_3
    var_5 = 'import os\nimport sys\nfrom pathlib import Path'
    var_6 = 'file2.py'
    var_7 = var_1 / var_6
    var_8 = 'import json\nfrom typing import List'
    var_9 = [var_1]
    var_10 = module_1.find_imports_in_paths(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 5
    var_13 = 'os'
    var_14 = 'sys'
    var_15 = 'pathlib'
    var_16 = 'json'
    var_17 = 'typing'
    var_18 = [var_1]
    var_19 = module_1.find_imports_in_paths(var_18, unique=var_2)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 4
    var_22 = 'file3.py'
    var_23 = var_1 / var_22
    var_24 = 'import os\n\ndef func():\n    import sys'
    var_25 = [var_1]
    var_26 = module_1.find_imports_in_paths(var_25, top_only=var_2)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 5
    var_29 = [var_1]
    var_30 = len(var_27)
    assert var_30 == 4



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = 'import os\nimport os'
    var_2 = True
    var_3 = 'import os\ndef foo():\n    import sys'
    var_4 = 'import os as operating_system\nimport os as os'
    var_5 = 0
    var_6 = 'import os.path\nimport os'
    var_7 = 'import os.path\nimport os.sys'
    var_8 = 'from os import path\nfrom os import sys'



# Parsed testcases at query #18
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
    var_8 = ''
    var_9 = 'import os\n'
    var_10 = 'import os\nimport sys\nimport json\n'
    var_11 = 'import os\nfrom sys import argv\nimport json\n'
    var_12 = 'import os\nfrom sys import argv\nimport json\nfrom os import path\n'



# Parsed testcases at query #19
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
    var_7 = 'import sys\nimport os\ninvalid syntax here\n'
    var_8 = module_0.check_file(var_7)
    var_9 = 'non_existent_file.py'
    var_10 = module_0.check_file(var_9)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = 'import sys\nimport os\n'
    var_4 = True
    var_5 = 0
    var_6 = 'import sys\nimport os\n'
    var_7 = 50



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = 'import sys\nimport os\n'
    var_2 = 0
    var_3 = True
    var_4 = 0
    var_5 = 0
    var_6 = True
    var_7 = True
    var_8 = True



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_2 = True
    var_3 = 'import os\n\ndef foo():\n    import sys\n'
    var_4 = 'non_existent.py'
    var_5 = list(var_0)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = 50
    var_8 = module_0.Config()



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = 'import os\nimport os\nfrom pathlib import Path\nfrom pathlib import Path'
    var_2 = True
    var_3 = 'import os\nimport os.path\nfrom pathlib import Path\nfrom pathlib import PurePath'
    var_4 = 'from os import path\nfrom os import sep\nfrom pathlib import Path\nfrom pathlib import PurePath'
    var_5 = 'import os\nimport os.path\nimport sys\nimport sys.platform'
    var_6 = 'import os\n\ndef foo():\n    import sys'
    var_7 = ''
    var_8 = 'import os\nimport sys'
    var_9 = module_0.Config()



# Parsed testcases at query #24
#--------------------------


import zipfile as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = module_0.Path(var_0)
    var_2 = True
    var_3 = 'file1.py'
    var_4 = var_1 / var_3
    var_5 = 'import os\nimport sys\nfrom pathlib import Path'
    var_6 = 'file2.py'
    var_7 = var_1 / var_6
    var_8 = 'import sys\nfrom pathlib import Path\nimport os'
    var_9 = [var_1]
    var_10 = module_1.find_imports_in_paths(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 6
    var_13 = [var_1]
    var_14 = module_1.find_imports_in_paths(var_13, unique=var_2)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = [var_1]
    var_18 = len(var_15)
    assert var_18 == 3
    var_19 = 'file3.py'
    var_20 = var_1 / var_19
    var_21 = 'import os\n\ndef foo():\n    import sys'
    var_22 = [var_1]
    var_23 = module_1.find_imports_in_paths(var_22, top_only=var_2)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 4



# Parsed testcases at query #25
#--------------------------


import isort.api as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    assert var_2 == 3
    var_3 = True
    var_4 = 'non_existent_file.py'
    var_5 = module_0.find_imports_in_file(var_4)
    var_6 = list(var_5)



# Parsed testcases at query #26
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
    var_11 = False
    var_12 = 'import sys\nimport os\n'
    var_13 = 'import os\nimport sys\ninvalid syntax here\n'
    var_14 = True
    var_15 = module_1.Config()
    var_16 = module_0.check_file(var_13, config=var_15)



# Parsed testcases at query #27
#--------------------------


import _io as module_0
import zipfile as module_1

def test_case_0():
    var_0 = 'import b\nimport a'
    var_1 = module_0.StringIO()
    var_2 = 'import a\nimport b'
    var_3 = module_0.StringIO()
    var_4 = module_0.StringIO()
    var_5 = True
    var_6 = module_0.StringIO()
    var_7 = 'test.py'
    var_8 = module_1.Path(var_7)
    var_9 = 'py'
    var_10 = module_0.StringIO()
    var_11 = 50
    var_12 = module_0.StringIO()
    var_13 = 'skip.py'
    var_14 = module_1.Path(var_13)
    var_15 = [var_13]
    var_16 = 'import b\nimport a\ninvalid syntax'
    var_17 = module_0.StringIO()
    var_18 = True



# Parsed testcases at query #28
#--------------------------


import _io as module_0
import isort.api as module_1

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = 'import b\nimport a\n'
    var_2 = True
    var_3 = 'import b\nimport a\n'
    var_4 = True
    var_5 = 'import b\nimport a\n'
    assert var_5 == 'import a\nimport b\n'
    var_6 = module_0.StringIO()
    var_7 = 'import a\nimport b\n'
    var_8 = 'non_existent_file.py'
    var_9 = module_1.sort_file(var_8)
    var_10 = 'import a\nif\nimport b\n'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_6 = 'py'
    var_7 = [var_4]
    var_8 = module_2.Config()
    var_9 = True
    var_10 = 120



# Parsed testcases at query #31
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
    var_8 = 'import os\nimport sys\ninvalid syntax here\n'
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
    var_20 = module_0.StringIO()
    var_21 = 'py'
    var_22 = module_0.StringIO()
    var_23 = 100



# Parsed testcases at query #32
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import b\nimport a\n'
    var_1 = 'import b\nimport a\n'
    var_2 = 'import b\nimport a\n'
    var_3 = True
    var_4 = 'import b\nimport a\n'
    var_5 = 50
    var_6 = module_0.Config()
    var_7 = 'import a\nimport b\n'



