####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os'
    var_2 = 'import os, sys'
    var_3 = module_0.strip_syntax(var_2)
    assert var_3 == 'os sys'
    var_4 = 'from os import path'
    var_5 = module_0.strip_syntax(var_4)
    assert var_5 == 'os path'
    var_6 = 'from os.path import (join, dirname)'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'os.path join dirname'
    var_8 = 'import os.path'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'os.path'
    var_10 = 'from . import foo'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == '. foo'
    var_12 = 'from .. import bar'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == '.. bar'
    var_14 = 'import os as o'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == 'os as o'
    var_16 = 'from os import path as p'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == 'os path as p'
    var_18 = 'import os\\'
    var_19 = module_0.strip_syntax(var_18)
    assert var_19 == 'os'
    var_20 = 'from os import (path, dirname)'
    var_21 = module_0.strip_syntax(var_20)
    assert var_21 == 'os path dirname'
    var_22 = 'import os, sys, json'
    var_23 = module_0.strip_syntax(var_22)
    assert var_23 == 'os sys json'
    var_24 = 'from os.path import join as j, dirname as d'
    var_25 = module_0.strip_syntax(var_24)
    assert var_25 == 'os.path join as j dirname as d'
    var_26 = 'import os._import'
    var_27 = module_0.strip_syntax(var_26)
    assert var_27 == 'os._import'
    var_28 = 'import os._cimport'
    var_29 = module_0.strip_syntax(var_28)
    assert var_29 == 'os._cimport'
    var_30 = 'from os import { path }'
    var_31 = module_0.strip_syntax(var_30)
    assert var_31 == 'os|path|'



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\nimport sys'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport os\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    sep\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import path,\nfrom sys import argv,\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'from os import path  # comment1\nfrom sys import argv  # comment2\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os\n'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)
    var_24 = 79
    var_25 = module_1.Config()
    var_26 = 'from os import path, sep\n'
    var_27 = module_0.file_contents(var_26, var_25)



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from module import (\n    thing1,\n    thing2,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = '# Just a comment\n# Another comment\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = var_17.lines_without_imports
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = "x = 1\nimport os\nprint('hello')\n"
    var_21 = module_0.file_contents(var_20)
    var_22 = var_21.lines_without_imports
    var_23 = len(var_22)
    assert var_23 == 2



# Parsed testcases at query #4
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom collections import defaultdict as dd\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os\n'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n# isort: imports-firstparty\nimport my_module\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from os import (\n    path,\n    sep,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from os import path  # comment for path\nfrom sys import argv  # comment for argv\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from os import path  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = '# noqa'
    var_36 = [var_35]
    var_37 = module_1.Config()
    var_38 = 'import os  # noqa\n# noqa\nimport sys\n'
    var_39 = module_0.file_contents(var_38, var_37)



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import pandas as pd'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'from .cimport something'
    var_13 = module_0.import_type(var_12)
    assert var_13 == 'from'
    var_14 = 'import os  # noqa'
    var_15 = True
    var_16 = module_1.Config()
    var_17 = module_0.import_type(var_14, var_16)
    assert var_17 is None
    var_18 = 'from os import path  # noqa'
    var_19 = module_1.Config()
    var_20 = module_0.import_type(var_18, var_19)
    assert var_20 is None
    var_21 = 'import os  # isort:skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'from os import path  # isort: skip'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = 'import os  # isort: split'
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = "print('hello')"
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = 'x = 5'
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = '# This is a comment'
    var_32 = module_0.import_type(var_31)
    assert var_32 is None
    var_33 = ''
    var_34 = module_0.import_type(var_33)
    assert var_34 is None
    var_35 = 'import*'
    var_36 = module_0.import_type(var_35)
    assert var_36 == 'straight'
    var_37 = 'import *'
    var_38 = module_0.import_type(var_37)
    assert var_38 == 'straight'
    var_39 = 'from .import module'
    var_40 = module_0.import_type(var_39)
    assert var_40 == 'from'
    var_41 = 'from . cimport module'
    var_42 = module_0.import_type(var_41)
    assert var_42 == 'from'



# Parsed testcases at query #7
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import (\n    path,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort:imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'import os\nimport sys\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os\n'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)



# Parsed testcases at query #8
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'x = 5'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #9
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = ''
    var_21 = module_0.file_contents(var_20)
    var_22 = '# isort: imports-thirdparty\nimport os\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    sep,\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from os import path  # comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os\n\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\nimport sys\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'import os'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'x = 1\nimport os\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = '# isort: imports-thirdparty\nimport os\n'
    var_37 = module_0.file_contents(var_36)



# Parsed testcases at query #10
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport os\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\r\nimport sys\r\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    sep,\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = "x = 1\nimport os\nprint('hello')\n"
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = '# Comment\nimport os  # inline comment\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = var_13.lines_without_imports
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = '# isort: imports-thirdparty\nimport os\n# isort: imports\nimport sys\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = 'from os import (\n    path,\n    sep,\n)\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)
    var_28 = 'import os\nimport sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = ''
    var_33 = module_0.file_contents(var_32)
    var_34 = var_33.imports
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = var_33.lines_without_imports
    var_37 = len(var_36)
    assert var_37 == 0
    var_38 = '# Just a comment\n# Another comment\n'
    var_39 = module_0.file_contents(var_38)
    var_40 = var_39.imports
    var_41 = len(var_40)
    assert var_41 == 0
    var_42 = var_39.lines_without_imports
    var_43 = len(var_42)
    assert var_43 == 2



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'x = 5'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = "x = 1\nimport os\nprint('hello')\n"
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = '# This is a comment\nimport os  # inline comment\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = var_13.lines_without_imports
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = '# isort: imports-thirdparty\nimport os\n# isort: imports-firstparty\nimport sys\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = var_17.lines_without_imports
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = var_21.lines_without_imports
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = 'from os import (\n    path,\n    sep,\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = 'from os import (  # comment for path\n    path,  # another comment\n    sep,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = True
    var_33 = module_1.Config()
    var_34 = 'import os\n'
    var_35 = module_0.file_contents(var_34, var_33)
    var_36 = var_35.verbose_output
    var_37 = len(var_36)
    var_38 = var_35.lines_without_imports
    var_39 = len(var_38)
    assert var_39 == 0
    var_40 = 79
    var_41 = module_1.Config()
    var_42 = 'from os import path, sep\n'
    var_43 = module_0.file_contents(var_42, var_41)
    var_44 = var_43.lines_without_imports
    var_45 = len(var_44)
    assert var_45 == 0
    var_46 = ''
    var_47 = module_0.file_contents(var_46)
    var_48 = var_47.imports
    var_49 = len(var_48)
    assert var_49 == 0
    var_50 = var_47.lines_without_imports
    var_51 = len(var_50)
    assert var_51 == 0
    var_52 = '# This is a comment\n# Another comment\n'
    var_53 = module_0.file_contents(var_52)
    var_54 = var_53.imports
    var_55 = len(var_54)
    assert var_55 == 0
    var_56 = var_53.lines_without_imports
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = 'from os import (\n    path,\n    sep,\n)\n'
    var_59 = module_0.file_contents(var_58)
    var_60 = var_59.lines_without_imports
    var_61 = len(var_60)
    assert var_61 == 0
    var_62 = 'from os import path, \\\n    sep\n'
    var_63 = module_0.file_contents(var_62)
    var_64 = var_63.lines_without_imports
    var_65 = len(var_64)
    assert var_65 == 0
    var_66 = 'import os; import sys\n'
    var_67 = module_0.file_contents(var_66)
    var_68 = var_67.lines_without_imports
    var_69 = len(var_68)
    assert var_69 == 0
    var_70 = 'import os  # isort: skip\nimport sys\n'
    var_71 = module_0.file_contents(var_70)
    var_72 = var_71.imports
    var_73 = len(var_72)
    assert var_73 == 0
    var_74 = var_71.lines_without_imports
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = 'import os\r\nimport sys\r\n'
    var_77 = module_0.file_contents(var_76)
    var_78 = var_77.lines_without_imports
    var_79 = len(var_78)
    assert var_79 == 0



# Parsed testcases at query #14
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = '# This is a comment\nimport os\n# Another comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import path\nfrom sys import argv\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    sep,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports-firstparty\nimport my_module\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'from os import (\n    path,\n    sep,\n)\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'from os import path  # comment for path\nfrom sys import argv  # comment for argv\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = 'x = 1\ny = 2\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 2



# Parsed testcases at query #15
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import Class'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os  # noqa'
    var_15 = module_0.import_type(var_14, var_13)
    assert var_15 is None
    var_16 = 'from os import path  # NOQA'
    var_17 = module_0.import_type(var_16, var_13)
    assert var_17 is None
    var_18 = 'import os  # isort:skip'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'from os import path  # isort: skip'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # isort: split'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = "print('hello')"
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = '# comment'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'x = 5'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = ''
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = '   '
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = 'import*'
    var_35 = module_0.import_type(var_34)
    assert var_35 is None



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from os import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'x = 1'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = "print('hello')"
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = module_0.import_type(var_8)
    assert var_25 == 'straight'



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport os\n# isort: imports\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    sep\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import path,\nfrom sys import argv,\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = var_23.lines_without_imports
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os  # isort:skip\nimport sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'from os import (  # comment1\n    path,  # comment2\n    sep  # comment3\n)\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = '# Above comment\nimport os\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = module_1.Config()
    var_37 = 'from os import path  # comment\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = module_1.Config()
    var_40 = 'import os as os\nfrom sys import argv as argv\n'
    var_41 = module_0.file_contents(var_40, var_39)
    var_42 = 'from module cimport func\n'
    var_43 = module_0.file_contents(var_42)
    var_44 = 'import os; import sys\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = 'from os import path, \\\n    sep\n'
    var_47 = module_0.file_contents(var_46)



# Parsed testcases at query #18
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import Class'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'import os  # NOQA'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = False
    var_20 = module_1.Config()
    var_21 = module_0.import_type(var_12, var_20)
    assert var_21 == 'straight'
    var_22 = 'import os  # isort:skip'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'import os  # isort: skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort: split'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'x = 1'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = "print('hello')"
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = ''
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = '  '
    var_35 = module_0.import_type(var_34)
    assert var_35 is None
    var_36 = 'import*'
    var_37 = module_0.import_type(var_36)
    assert var_37 is None
    var_38 = 'fromimport'
    var_39 = module_0.import_type(var_38)
    assert var_39 is None
    var_40 = 'import os # comment'
    var_41 = module_0.import_type(var_40)
    assert var_41 == 'straight'
    var_42 = 'from os import path # comment'
    var_43 = module_0.import_type(var_42)
    assert var_43 == 'from'



# Parsed testcases at query #19
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# isort: imports-firstparty\nimport my_module'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n)'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)



# Parsed testcases at query #20
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport os'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os as operating_system'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,  # comment\n)'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import os\n\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import os\nimport sys'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #21
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import List'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import defaultdict as dd'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from typing import (  # comment1\n    List,  # comment2\n    Dict,  # comment3\n)'
    var_15 = module_0.file_contents(var_14)
    var_16 = '# Above comment\nimport os'
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)
    var_24 = 'import os\nimport sys'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import os\r\nimport sys'
    var_27 = module_0.file_contents(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    var_30 = '# Just a comment'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'from typing import (\n    List,\n    Dict\n)'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'from typing import \\\n    List'
    var_35 = module_0.file_contents(var_34)
    var_36 = 'import os; import sys'
    var_37 = module_0.file_contents(var_36)
    var_38 = '# isort: skip\nimport os'
    var_39 = module_0.file_contents(var_38)
    var_40 = module_1.Config()
    var_41 = 'from typing import List  # comment'
    var_42 = module_0.file_contents(var_41, var_40)
    var_43 = module_1.Config()
    var_44 = 'import os as os'
    var_45 = module_0.file_contents(var_44, var_43)



# Parsed testcases at query #22
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import  os'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = 'import os  # isort:skip'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'from os import path  # isort: skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'import os  # isort: split'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = 'x = 1'
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = "print('hello')"
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = ''
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = 'import*'
    var_32 = module_0.import_type(var_31)
    assert var_32 is None
    var_33 = 'fromimport os'
    var_34 = module_0.import_type(var_33)
    assert var_34 is None



# Parsed testcases at query #23
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'import os  # NOQA'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = False
    var_20 = module_1.Config()
    var_21 = module_0.import_type(var_12, var_20)
    assert var_21 == 'straight'
    var_22 = 'import os  # isort:skip'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'import os  # isort: skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort: split'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = "print('hello')"
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = 'x = 1'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = ''
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = '  '
    var_35 = module_0.import_type(var_34)
    assert var_35 is None
    var_36 = 'fromimport os'
    var_37 = module_0.import_type(var_36)
    assert var_37 is None
    var_38 = 'importfrom os'
    var_39 = module_0.import_type(var_38)
    assert var_39 is None



# Parsed testcases at query #24
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = var_15.lines_without_imports
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = 'x = 1\ny = 2\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = var_19.lines_without_imports
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'import os\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import os\r\nimport sys\r\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = True
    var_27 = module_1.Config()
    var_28 = 'import os\n'
    var_29 = module_0.file_contents(var_28, var_27)
    var_30 = var_29.verbose_output
    var_31 = len(var_30)



# Parsed testcases at query #25
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 'import os\nx = 1\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = '# Comment\nimport os\n# Another comment\nimport sys\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    sep,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = '# isort: imports-thirdparty\nimport os\n# isort: imports-firstparty\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = True
    var_21 = module_1.Config()
    var_22 = 'import os\n'
    var_23 = module_0.file_contents(var_22, var_21)
    var_24 = var_23.verbose_output
    var_25 = len(var_24)
    var_26 = 'import os\r\nimport sys\r\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = var_29.imports
    var_33 = len(var_32)
    assert var_33 == 0
    var_34 = 'from os import (\n    path,\n    sep\n)\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = 'from os import path  # comment for path\nfrom sys import argv  # comment for argv\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = 'import os\nimport sys\nx = 1\n'
    var_39 = module_0.file_contents(var_38)



# Parsed testcases at query #26
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# This is a comment\nimport os  # inline comment\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports\nimport os\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import OrderedDict as OD\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    # comment\n    sys,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# comment 1\n# comment 2\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = var_27.lines_without_imports
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from typing import Any  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = '# noqa'
    var_36 = [var_35]
    var_37 = module_1.Config()
    var_38 = 'import os  # noqa\n# comment\nimport sys\n'
    var_39 = module_0.file_contents(var_38, var_37)
    var_40 = module_1.Config()
    var_41 = 'import os as os\nfrom typing import Dict as Dict\n'
    var_42 = module_0.file_contents(var_41, var_40)
    var_43 = module_1.Config()
    var_44 = 'from typing import Dict as D, List as L\n'
    var_45 = module_0.file_contents(var_44, var_43)



# Parsed testcases at query #27
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import List\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from typing import (\n    List,\n    Dict,\n    Optional,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os\n'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)



# Parsed testcases at query #28
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport requests'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from os import (\n    path,\n)'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = '# Just a comment'
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 1
    var_28 = '# isort: skip\nimport os\nimport sys'
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = 'from typing import List, Dict'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'from os import (\n    path,  # path comment\n    sep,  # sep comment\n)'
    var_35 = module_0.file_contents(var_34)
    var_36 = module_1.Config()
    var_37 = 'from os import path  # comment'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = '# noqa'
    var_40 = [var_39]
    var_41 = module_1.Config()
    var_42 = 'import os  # noqa'
    var_43 = module_0.file_contents(var_42, var_41)
    var_44 = var_43.lines_without_imports
    var_45 = len(var_44)
    assert var_45 == 1



# Parsed testcases at query #29
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'import os  # NOQA'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = 'import os  # isort:skip'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'import os  # isort: skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'import os  # isort: split'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = 'x = 1'
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = "print('hello')"
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = ''
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = '  '
    var_32 = module_0.import_type(var_31)
    assert var_32 is None
    var_33 = module_0.import_type(var_12)
    assert var_33 == 'straight'



# Parsed testcases at query #30
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n# Another comment\nimport sys\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport os\n# isort: imports-firstparty\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    sep,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,  # path comment\n    sep,  # sep comment\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from os import (\n    path,\n    sep\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os  # isort: skip\nimport sys\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from os import path  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = '# noqa'
    var_36 = [var_35]
    var_37 = module_1.Config()
    var_38 = 'import os  # noqa\n# noqa\nimport sys\n'
    var_39 = module_0.file_contents(var_38, var_37)



# Parsed testcases at query #31
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport numpy\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\r\nimport sys\r\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = ''
    var_21 = module_0.file_contents(var_20)
    var_22 = 'from os import (\n    path,\n    sep\n)\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_25 = module_0.file_contents(var_24)



# Parsed testcases at query #32
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    sep,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (  # comment for path\n    path,  # another comment\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\n'
    var_23 = '\r\n'
    var_24 = module_1.Config()
    var_25 = module_0.file_contents(var_22, var_24)
    var_26 = ''
    var_27 = module_0.file_contents(var_26)
    var_28 = '# Just a comment\n# Another comment\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = 'from os import (\n    path,\n    sep\n)\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'from os import path, \\\n    sep\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = 'import os; import sys\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = 'import os  # isort:skip\nimport sys\n'
    var_39 = module_0.file_contents(var_38)
    var_40 = 'import os  # "comment"\nimport sys  # \'comment\'\n'
    var_41 = module_0.file_contents(var_40)



# Parsed testcases at query #33
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import Class'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = False
    var_20 = module_1.Config()
    var_21 = module_0.import_type(var_12, var_20)
    assert var_21 == 'straight'
    var_22 = 'import os  # isort:skip'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'from os import path  # isort: skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort: split'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = "print('hello')"
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = 'x = 5'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = '# This is a comment'
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = ''
    var_35 = module_0.import_type(var_34)
    assert var_35 is None
    var_36 = 'import*'
    var_37 = module_0.import_type(var_36)
    assert var_37 is None
    var_38 = 'fromimport os'
    var_39 = module_0.import_type(var_38)
    assert var_39 is None
    var_40 = 'import os.path'
    var_41 = module_0.import_type(var_40)
    assert var_41 == 'straight'
    var_42 = 'from . import *'
    var_43 = module_0.import_type(var_42)
    assert var_43 == 'from'



# Parsed testcases at query #34
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = 'import os  # isort:skip'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'from os import path  # isort: skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'import os  # isort: split'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = "print('hello')"
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = '# comment'
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = ''
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = 'import*'
    var_32 = module_0.import_type(var_31)
    assert var_32 is None
    var_33 = 'fromimport os'
    var_34 = module_0.import_type(var_33)
    assert var_34 is None



# Parsed testcases at query #35
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# This is a comment\nimport os  # inline comment\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from typing import (\n    Any,  # comment1\n    Dict,  # comment2\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from typing import \\\n    Any, \\\n    Dict\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from typing import Any  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = module_1.Config()
    var_36 = 'import os as os\nfrom typing import Dict as Dict\n'
    var_37 = module_0.file_contents(var_36, var_35)
    var_38 = '# noqa'
    var_39 = [var_38]
    var_40 = module_1.Config()
    var_41 = 'import os  # noqa\n# noqa\nimport sys\n'
    var_42 = module_0.file_contents(var_41, var_40)



# Parsed testcases at query #36
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# Comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import (\n    path,\n    sep,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from os import (\n    path,\n    sep\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\n\n\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)



# Parsed testcases at query #37
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from typing import (\n    Any,  # comment1\n    Dict,  # comment2\n)\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'import os\n"""docstring"""\nimport sys\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'from module cimport Class\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = 'import os as os\n'
    var_37 = module_1.Config()
    var_38 = module_0.file_contents(var_36, var_37)
    var_39 = 'from typing import Any  # comment\n'
    var_40 = module_1.Config()
    var_41 = module_0.file_contents(var_39, var_40)



# Parsed testcases at query #38
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    sys,\n)'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from typing import (\n    List,\n    Dict,\n    Set,\n)'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from typing import (\n    List, \\\n    Dict,\n)'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'import os; import sys'
    var_33 = module_0.file_contents(var_32)
    var_34 = module_1.Config()
    var_35 = 'import os as os'
    var_36 = module_0.file_contents(var_35, var_34)
    var_37 = 'from module cimport func'
    var_38 = module_0.file_contents(var_37)



# Parsed testcases at query #39
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import numpy as np\nimport pandas as pd'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import path as p\nfrom sys import argv as a'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# This is a comment\nimport os  # inline comment'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    environ\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport os'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = '# Comment 1\n# Comment 2'
    var_17 = module_0.file_contents(var_16)
    var_18 = var_17.lines_without_imports
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'x = 1\nimport os\ny = 2'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)



# Parsed testcases at query #40
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import (\n    path,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-firstparty\nimport my_module\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = ''
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\nx = 1\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    # comment\n    environ,\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = module_1.Config()
    var_27 = 'from os import path  # comment\n'
    var_28 = module_0.file_contents(var_27, var_26)
    var_29 = '# isort: skip\nimport os\nimport sys\n'
    var_30 = module_0.file_contents(var_29)
    var_31 = '# noqa'
    var_32 = [var_31]
    var_33 = module_1.Config()
    var_34 = '# noqa\nimport os\n'
    var_35 = module_0.file_contents(var_34, var_33)
    var_36 = '# ---'
    var_37 = [var_36]
    var_38 = module_1.Config()
    var_39 = '# ---\nimport os\n'
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = module_1.Config()
    var_42 = 'x = 1\nimport os\n'
    var_43 = module_0.file_contents(var_42, var_41)
    var_44 = module_1.Config()
    var_45 = 'import numpy as numpy\n'
    var_46 = module_0.file_contents(var_45, var_44)
    var_47 = module_1.Config()
    var_48 = 'from pandas import DataFrame as DF, Series as S\n'
    var_49 = module_0.file_contents(var_48, var_47)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os'
    var_2 = 'import (os, sys)'
    var_3 = module_0.strip_syntax(var_2)
    assert var_3 == 'os sys'
    var_4 = 'import os,\\ sys'
    var_5 = module_0.strip_syntax(var_4)
    assert var_5 == 'os sys'
    var_6 = 'from os import path'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'os path'
    var_8 = 'from os import (path, walk)'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'os path walk'
    var_10 = 'cimport os'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == 'os'
    var_12 = 'import _import'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == '_import'
    var_14 = 'cimport _cimport'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == '_cimport'
    var_16 = 'from . import { a as b, c }'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == '. {|a as b| c|}'



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import defaultdict as dd\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    sys,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = True
    var_21 = module_1.Config()
    var_22 = 'import os\n'
    var_23 = module_0.file_contents(var_22, var_21)
    var_24 = var_23.verbose_output
    var_25 = len(var_24)



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom collections import OrderedDict as OD\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from os import (\n    path,\n    sys,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = "import os\n'''\ndocstring\n'''\nimport sys\n"
    var_29 = module_0.file_contents(var_28)
    var_30 = module_1.Config()
    var_31 = 'from typing import Any  # comment\n'
    var_32 = module_0.file_contents(var_31, var_30)
    var_33 = '# noqa'
    var_34 = [var_33]
    var_35 = module_1.Config()
    var_36 = 'import os  # noqa\nimport sys\n'
    var_37 = module_0.file_contents(var_36, var_35)
    var_38 = module_1.Config()
    var_39 = 'import numpy as np\nfrom os import path as path\n'
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = module_1.Config()
    var_42 = 'from typing import OrderedDict as OD, List as L\n'
    var_43 = module_0.file_contents(var_42, var_41)



# Parsed testcases at query #4
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os'
    var_2 = 'import os, sys'
    var_3 = module_0.strip_syntax(var_2)
    assert var_3 == 'os sys'
    var_4 = 'from os import path'
    var_5 = module_0.strip_syntax(var_4)
    assert var_5 == 'os path'
    var_6 = 'from os import (path, join)'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'os path join'
    var_8 = 'import os.path'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'os.path'
    var_10 = 'from . import os'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == '. os'
    var_12 = 'from .. import os'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == '.. os'
    var_14 = 'import os as operating_system'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == 'os as operating_system'
    var_16 = 'from os import path as p'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == 'os path as p'
    var_18 = 'import os\\'
    var_19 = module_0.strip_syntax(var_18)
    assert var_19 == 'os'
    var_20 = 'import os, \\'
    var_21 = module_0.strip_syntax(var_20)
    assert var_21 == 'os'
    var_22 = 'from os import (path, \\'
    var_23 = module_0.strip_syntax(var_22)
    assert var_23 == 'os path'
    var_24 = 'import _import'
    var_25 = module_0.strip_syntax(var_24)
    assert var_25 == '_import'
    var_26 = 'import _cimport'
    var_27 = module_0.strip_syntax(var_26)
    assert var_27 == '_cimport'
    var_28 = 'from os import { path, join }'
    var_29 = module_0.strip_syntax(var_28)
    assert var_29 == 'os {|path| |join| |}'



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n    Set,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import defaultdict as dd\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = var_15.lines_without_imports
    var_17 = len(var_16)
    assert var_17 == 0
    var_18 = 'x = 1\ny = 2\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = var_19.lines_without_imports
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)
    var_28 = 'import os\nimport sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'numpy'
    var_33 = [var_32]
    var_34 = module_1.Config()
    var_35 = 'import numpy\nimport pandas\n'
    var_36 = module_0.file_contents(var_35, var_34)
    var_37 = 'import os  # isort:skip\nimport sys\n'
    var_38 = module_0.file_contents(var_37)
    var_39 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_40 = module_0.file_contents(var_39)
    var_41 = '# Above comment\nimport os\n'
    var_42 = module_0.file_contents(var_41)
    var_43 = module_1.Config()
    var_44 = 'x = 1\nimport os\n'
    var_45 = module_0.file_contents(var_44, var_43)
    var_46 = module_1.Config()
    var_47 = 'import numpy as np\nfrom numpy import array as array\n'
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = module_1.Config()
    var_50 = 'from typing import List as L, Dict as D\n'
    var_51 = module_0.file_contents(var_50, var_49)
    var_52 = '# noqa'
    var_53 = [var_52]
    var_54 = module_1.Config()
    var_55 = '# noqa\nimport os\n'
    var_56 = module_0.file_contents(var_55, var_54)
    var_57 = module_1.Config()
    var_58 = '# comment\nimport os\n'
    var_59 = module_0.file_contents(var_58, var_57)
    var_60 = module_1.Config()
    var_61 = 'from typing import List  # comment\n'
    var_62 = module_0.file_contents(var_61, var_60)
    var_63 = 'FUTURE'
    var_64 = 'STDLIB'
    var_65 = 'THIRDPARTY'
    var_66 = [var_63, var_64, var_65]
    var_67 = module_1.Config()
    var_68 = 'import unknown_module\n'
    var_69 = module_0.file_contents(var_68, var_67)
    var_70 = 'from module cimport func\n'
    var_71 = module_0.file_contents(var_70)
    var_72 = 'import os; import sys\n'
    var_73 = module_0.file_contents(var_72)
    var_74 = 'from typing import \\\n    List\n'
    var_75 = module_0.file_contents(var_74)
    var_76 = module_1.Config()
    var_77 = 'import os as os\nfrom os import path as path\n'
    var_78 = module_0.file_contents(var_77, var_76)



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import exit'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'x = 1'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = "print('hello')"
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #7
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = '# comment'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #8
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import exit'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import something'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = True
    var_9 = module_1.Config()
    var_10 = 'import os  # noqa'
    var_11 = module_0.import_type(var_10, var_9)
    assert var_11 is None
    var_12 = 'from sys import exit  # NOQA'
    var_13 = module_0.import_type(var_12, var_9)
    assert var_13 is None
    var_14 = 'import os  # isort:skip'
    var_15 = module_0.import_type(var_14)
    assert var_15 is None
    var_16 = 'from sys import exit  # isort: skip'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = 'import os  # isort: split'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = "print('hello')"
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'x = 1'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = ''
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'fromimport os'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'import* os'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None



# Parsed testcases at query #9
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 'import os\nx = 1\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = '# Comment\nimport os  # inline comment\n# Another comment\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = var_13.lines_without_imports
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'from os import (\n    path,\n    curdir,\n)\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = '# isort:imports-thirdparty\nimport os\n# isort: imports-local\nimport local_module\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)
    var_28 = 'import os\n\n\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = ''
    var_33 = module_0.file_contents(var_32)
    var_34 = var_33.lines_without_imports
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = 'from os import (\n    path,\n    curdir\n)\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = 'from os import path  # comment for path\nfrom sys import argv  # comment for argv\n'
    var_39 = module_0.file_contents(var_38)



# Parsed testcases at query #10
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport os\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (  # comment\n    path,  # path comment\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = '# Just a comment\n# Another comment\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from os import (\n    path,\n    sep,\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os; import sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from os import path, \\\n    sep\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from os import path  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = module_1.Config()
    var_36 = 'import os as os\nfrom sys import argv as argv\n'
    var_37 = module_0.file_contents(var_36, var_35)



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'import os\nx = 1\nimport sys\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# Comment\nimport os  # inline comment\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport os\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from os import (\n    path,\n    curdir,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'from os import (  # comment1\n    path,  # comment2\n    curdir,  # comment3\n)\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = True
    var_19 = module_1.Config()
    var_20 = 'import os\n'
    var_21 = module_0.file_contents(var_20, var_19)
    var_22 = var_21.verbose_output
    var_23 = len(var_22)
    var_24 = 'import os\n\n\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import os\r\nimport sys\r\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = '# This is a comment\nimport os  # inline comment\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = ''
    var_19 = module_0.file_contents(var_18)
    var_20 = var_19.lines_without_imports
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "import 'os'"
    var_6 = ''
    var_7 = 0
    var_8 = ()
    var_9 = module_0.skip_line(var_5, var_6, var_7, var_8)
    var_10 = 'import "os"'
    var_11 = ''
    var_12 = 0
    var_13 = ()
    var_14 = module_0.skip_line(var_10, var_11, var_12, var_13)
    var_15 = 'import """os"""'
    var_16 = ''
    var_17 = 0
    var_18 = ()
    var_19 = module_0.skip_line(var_15, var_16, var_17, var_18)
    var_20 = "import '''os'''"
    var_21 = ''
    var_22 = 0
    var_23 = ()
    var_24 = module_0.skip_line(var_20, var_21, var_22, var_23)
    var_25 = 'x = 1; import os'
    var_26 = ''
    var_27 = 0
    var_28 = ()
    var_29 = module_0.skip_line(var_25, var_26, var_27, var_28)
    var_30 = 'import os; import sys'
    var_31 = ''
    var_32 = 0
    var_33 = ()
    var_34 = module_0.skip_line(var_30, var_31, var_32, var_33)
    var_35 = 'import os; # comment'
    var_36 = ''
    var_37 = 0
    var_38 = ()
    var_39 = module_0.skip_line(var_35, var_36, var_37, var_38)
    var_40 = 'x = 1; # comment'
    var_41 = ''
    var_42 = 0
    var_43 = ()
    var_44 = module_0.skip_line(var_40, var_41, var_42, var_43)
    var_45 = 'import "os\\""'
    var_46 = ''
    var_47 = 0
    var_48 = ()
    var_49 = module_0.skip_line(var_45, var_46, var_47, var_48)
    var_50 = 'import os'
    var_51 = "'"
    var_52 = 0
    var_53 = ()
    var_54 = module_0.skip_line(var_50, var_51, var_52, var_53)
    var_55 = "import os'"
    var_56 = "'"
    var_57 = 0
    var_58 = ()
    var_59 = module_0.skip_line(var_55, var_56, var_57, var_58)
    var_60 = 'x = 1; import os'
    var_61 = ''
    var_62 = 0
    var_63 = ()
    var_64 = False
    var_65 = module_0.skip_line(var_60, var_61, var_62, var_63, var_64)



# Parsed testcases at query #14
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from collections import defaultdict\nfrom os import path\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-firstparty\nimport my_module\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\n\nx = 1\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\nimport sys\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = var_23.lines_without_imports
    var_25 = len(var_24)
    assert var_25 == 0
    var_26 = 'from os import (\n    path,\n    sep\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os as operating_system\nfrom collections import defaultdict as dd\n'
    var_29 = module_0.file_contents(var_28)



# Parsed testcases at query #15
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = "x = 1\nimport os\nprint('hello')\n"
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from typing import (\n    List,\n    Dict,\n    Any,\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = '# Above comment\nimport os\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from typing import List  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = module_1.Config()
    var_36 = 'import os as os\nfrom typing import List as List\n'
    var_37 = module_0.file_contents(var_36, var_35)
    var_38 = module_1.Config()
    var_39 = 'from pandas import DataFrame as DF, Series as S\n'
    var_40 = module_0.file_contents(var_39, var_38)



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport requests\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import os\r\nimport sys\r\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = ''
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from os import (\n    path,\n    pathsep,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\r\nimport sys\r\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    pathsep\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import numpy as np\nfrom os import path as osp\n'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #18
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'x = 5'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = False
    var_28 = module_1.Config()
    var_29 = module_0.import_type(var_8, var_28)
    assert var_29 == 'straight'



# Parsed testcases at query #19
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import submodule'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = 'import os  # isort:skip'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'from os import path  # isort: skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'import os  # isort: split'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = "print('hello')"
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = 'x = 1'
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = ''
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = False
    var_32 = module_1.Config()
    var_33 = module_0.import_type(var_12, var_32)
    assert var_33 == 'straight'



# Parsed testcases at query #20
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport numpy\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\r\nimport sys\r\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    sep\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import numpy as np\nfrom os import path as osp\n'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #21
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import exit'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'x = 5'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #22
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = "x = 1\nimport os\nprint('hello')\n"
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = '# This is a comment\nimport os  # inline comment\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = var_13.lines_without_imports
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from os import (\n    path,\n    environ\n)\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = var_17.lines_without_imports
    var_19 = len(var_18)
    assert var_19 == 0
    var_20 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = var_21.lines_without_imports
    var_23 = len(var_22)
    assert var_23 == 0
    var_24 = '# isort: imports-thirdparty\nimport os\n# isort: imports-firstparty\nimport sys\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = 'from os import path, environ,\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = True
    var_33 = module_1.Config()
    var_34 = 'import os\n'
    var_35 = module_0.file_contents(var_34, var_33)
    var_36 = var_35.verbose_output
    var_37 = len(var_36)
    var_38 = var_35.lines_without_imports
    var_39 = len(var_38)
    assert var_39 == 0
    var_40 = 'import os\r\nimport sys\r\n'
    var_41 = module_0.file_contents(var_40)
    var_42 = var_41.lines_without_imports
    var_43 = len(var_42)
    assert var_43 == 0



# Parsed testcases at query #23
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport os'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from os import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'import os  # noqa'
    var_7 = True
    var_8 = module_1.Config()
    var_9 = module_0.import_type(var_6, var_8)
    assert var_9 is None
    var_10 = 'import os  # NOQA'
    var_11 = module_1.Config()
    var_12 = module_0.import_type(var_10, var_11)
    assert var_12 is None
    var_13 = 'import os  # isort:skip'
    var_14 = module_0.import_type(var_13)
    assert var_14 is None
    var_15 = 'import os  # isort: skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: split'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'x = 1'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = ''
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = module_0.import_type(var_6)
    assert var_25 == 'straight'



# Parsed testcases at query #24
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import argv'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'x = 1'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = "print('hello')"
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



# Parsed testcases at query #25
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# This is a comment\nimport os  # inline comment\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport requests\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from typing import (\n    List,\n    Dict,\n    Set,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os  # isort: skip\nimport sys\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'from typing import List  # comment\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = module_1.Config()
    var_36 = 'import numpy as np\nfrom pandas import DataFrame as DataFrame\n'
    var_37 = module_0.file_contents(var_36, var_35)
    var_38 = module_1.Config()
    var_39 = 'from typing import List, Dict\n'
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = '# noqa'
    var_42 = [var_41]
    var_43 = module_1.Config()
    var_44 = 'import os  # noqa\n# noqa\nimport sys\n'
    var_45 = module_0.file_contents(var_44, var_43)



# Parsed testcases at query #26
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from os import (\n    path,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort:imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import os\nimport sys\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os\r\nimport sys\r\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    environ,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = ''
    var_19 = module_0.file_contents(var_18)
    var_20 = '# Just a comment\n# Another comment\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'x = 1\nimport os\nprint(x)\n'
    var_23 = module_0.file_contents(var_22)



# Parsed testcases at query #27
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# This is a comment\nimport os  # inline comment\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort: imports-thirdparty\nimport requests\n# isort: imports\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import os\r\nimport sys\r\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = '# Comment 1\n# Comment 2\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = var_33.lines_without_imports
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'from typing import Any  # Comment for Any\n'
    var_37 = module_0.file_contents(var_36)



# Parsed testcases at query #28
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nfrom sys import argv\nimport sys'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os\n# Another comment\nfrom sys import argv'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\nimport sys'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import os as operating_system\nfrom sys import argv as arguments'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort: imports-thirdparty\nimport os\n# isort: imports-firstparty\nimport sys'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = ''
    var_21 = module_0.file_contents(var_20)
    var_22 = '# Just a comment\n# Another comment'
    var_23 = module_0.file_contents(var_22)
    var_24 = var_23.lines_without_imports
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'from os import (\n    path,\n    sep,\n)\nimport sys'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #29
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'import os  # comment\n# comment above\nimport sys\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    sep,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import os\nimport sys\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import os\r\nimport sys\r\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'import os\nimport sys\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = '# isort: imports-thirdparty\nimport os\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from os import (\n    path,\n    sep\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os  # isort: skip\nimport sys\n'
    var_29 = module_0.file_contents(var_28)



# Parsed testcases at query #30
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import List'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = '# This is a comment\nimport os  # inline comment'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport numpy'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from typing import (\n    List,\n    Dict,\n)'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'import os'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = var_17.verbose_output
    var_19 = len(var_18)
    var_20 = 'import os\nimport sys'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'import os\r\nimport sys'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment'
    var_27 = module_0.file_contents(var_26)
    var_28 = var_27.lines_without_imports
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from typing import (\n    List,\n    Dict,\n    Set,\n)'
    var_31 = module_0.file_contents(var_30)



# Parsed testcases at query #31
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import defaultdict as dd\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\nimport sys\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)



# Parsed testcases at query #32
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from os import (\n    path,\n    curdir,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-firstparty\nimport my_module\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\r\nimport sys\r\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = ''
    var_23 = module_0.file_contents(var_22)
    var_24 = 'from os import (\n    path,\n    curdir\n)\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_27 = module_0.file_contents(var_26)



# Parsed testcases at query #33
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 0
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = var_5.lines_without_imports
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = 'import os\nx = 1\nimport sys\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = var_9.lines_without_imports
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = '# Comment\nimport os\n# Another comment\nimport sys\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = var_13.lines_without_imports
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'from os import (\n    path,\n)\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = 'import os as operating_system\nfrom sys import argv as arguments\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports\nimport os\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)
    var_28 = 'import os\nimport sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\nimport sys\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = ''
    var_33 = module_0.file_contents(var_32)
    var_34 = var_33.lines_without_imports
    var_35 = len(var_34)
    assert var_35 == 0
    var_36 = var_33.imports
    var_37 = len(var_36)
    assert var_37 == 0



# Parsed testcases at query #34
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'from os import (\n    path,\n)\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    Dict,\n    List,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,  # path comment\n    sep,  # sep comment\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'import os\nimport sys\n'
    var_17 = module_0.file_contents(var_16)
    var_18 = ''
    var_19 = module_0.file_contents(var_18)
    var_20 = 'def foo():\n    pass\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = var_25.verbose_output
    var_27 = len(var_26)
    var_28 = 'import os\n\n\n'
    var_29 = module_0.file_contents(var_28)



# Parsed testcases at query #35
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "import 'os'"
    var_6 = ()
    var_7 = module_0.skip_line(var_5, var_1, var_2, var_6)
    var_8 = 'import "os"'
    var_9 = ()
    var_10 = module_0.skip_line(var_8, var_1, var_2, var_9)
    var_11 = "import '''os'''"
    var_12 = ()
    var_13 = module_0.skip_line(var_11, var_1, var_2, var_12)
    var_14 = 'import """os"""'
    var_15 = ()
    var_16 = module_0.skip_line(var_14, var_1, var_2, var_15)
    var_17 = 'x = 1; import os'
    var_18 = ()
    var_19 = module_0.skip_line(var_17, var_1, var_2, var_18)
    var_20 = 'import os; import sys'
    var_21 = ()
    var_22 = module_0.skip_line(var_20, var_1, var_2, var_21)
    var_23 = 'x = 1; # comment'
    var_24 = ()
    var_25 = module_0.skip_line(var_23, var_1, var_2, var_24)
    var_26 = 'import "os\\"'
    var_27 = ()
    var_28 = module_0.skip_line(var_26, var_1, var_2, var_27)
    var_29 = 'import "os\''
    var_30 = ()
    var_31 = module_0.skip_line(var_29, var_1, var_2, var_30)
    var_32 = "'"
    var_33 = ()
    var_34 = module_0.skip_line(var_0, var_32, var_2, var_33)
    var_35 = "import os'"
    var_36 = ()
    var_37 = module_0.skip_line(var_35, var_32, var_2, var_36)
    var_38 = 'import os"""'
    var_39 = '"""'
    var_40 = ()
    var_41 = module_0.skip_line(var_38, var_39, var_2, var_40)
    var_42 = ()
    var_43 = False
    var_44 = module_0.skip_line(var_17, var_1, var_2, var_42, var_43)
    var_45 = 'x = 1; y = 2'
    var_46 = ()
    var_47 = False
    var_48 = module_0.skip_line(var_45, var_1, var_43, var_46, var_47)
    var_49 = ()
    var_50 = False
    var_51 = module_0.skip_line(var_23, var_1, var_47, var_49, var_50)
    var_52 = ()
    var_53 = False
    var_54 = module_0.skip_line(var_20, var_1, var_50, var_52, var_53)
    var_55 = ()
    var_56 = False
    var_57 = module_0.skip_line(var_17, var_1, var_53, var_55, var_56)



# Parsed testcases at query #36
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import OrderedDict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# Comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from typing import (\n    Any,\n    Dict,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports\nimport os\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os\n'
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = var_15.verbose_output
    var_17 = len(var_16)
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.file_contents(var_18)
    var_20 = 'import os\r\nimport sys\r\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = 'from typing import (\n    Any,\n    Dict,\n    List,\n)\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = 'import numpy as np\nfrom collections import OrderedDict as OD\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = 'from typing import (\n    Any,  # comment1\n    Dict,  # comment2\n)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.lines_without_imports
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = '# Just a comment\n# Another comment\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'import os  # isort: skip\nimport sys\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = module_1.Config()
    var_37 = 'from typing import Any  # comment\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = 'from module cimport func\n'
    var_40 = module_0.file_contents(var_39)
    var_41 = 'import os; import sys\n'
    var_42 = module_0.file_contents(var_41)
    var_43 = 'from typing import (\n    Any, \\\n    Dict\n)\n'
    var_44 = module_0.file_contents(var_43)



# Parsed testcases at query #37
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'import os  # noqa'
    var_13 = True
    var_14 = module_1.Config()
    var_15 = module_0.import_type(var_12, var_14)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_1.Config()
    var_18 = module_0.import_type(var_16, var_17)
    assert var_18 is None
    var_19 = 'import os  # isort:skip'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = 'from os import path  # isort: skip'
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'import os  # isort: split'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = "print('hello')"
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = '# just a comment'
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = ''
    var_30 = module_0.import_type(var_29)
    assert var_30 is None
    var_31 = 'import*'
    var_32 = module_0.import_type(var_31)
    assert var_32 is None
    var_33 = 'fromimport something'
    var_34 = module_0.import_type(var_33)
    assert var_34 is None



# Parsed testcases at query #38
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os, sys'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..module import something'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os  # noqa'
    var_15 = module_0.import_type(var_14, var_13)
    assert var_15 is None
    var_16 = 'from os import path  # NOQA'
    var_17 = module_0.import_type(var_16, var_13)
    assert var_17 is None
    var_18 = 'import os  # isort:skip'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'from os import path  # isort: skip'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # isort: split'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = "print('hello')"
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = '# This is a comment'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'x = 5'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = ''
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = '   '
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = 'import*'
    var_35 = module_0.import_type(var_34)
    assert var_35 is None



# Parsed testcases at query #39
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import exit'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = True
    var_9 = module_1.Config()
    var_10 = 'import os  # noqa'
    var_11 = module_0.import_type(var_10, var_9)
    assert var_11 is None
    var_12 = 'from sys import exit  # NOQA'
    var_13 = module_0.import_type(var_12, var_9)
    assert var_13 is None
    var_14 = 'import os  # isort:skip'
    var_15 = module_0.import_type(var_14)
    assert var_15 is None
    var_16 = 'from sys import exit  # isort: skip'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = 'import os  # isort: split'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = "print('hello')"
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'x = 5'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = ''
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import*'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'fromimport sys'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None



# Parsed testcases at query #40
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import Any\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os\nx = 1\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = '# This is a comment\nimport os  # inline comment\n# Another comment\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = '# isort: imports-thirdparty\nimport numpy\n# isort: imports-firstparty\nimport my_module\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'import numpy as np\nfrom collections import defaultdict as dd\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'from os import (\n    path,\n    sys,\n)\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = True
    var_17 = module_1.Config()
    var_18 = 'import os\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = var_19.verbose_output
    var_21 = len(var_20)
    var_22 = 'import os\r\nimport sys\r\n'
    var_23 = module_0.file_contents(var_22)
    var_24 = ''
    var_25 = module_0.file_contents(var_24)
    var_26 = '# Just a comment\n# Another comment\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from typing import (\n    List,\n    Dict,\n    Any,\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from typing import (\n    List, \\\n    Dict,\n)\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = 'import os; import sys\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = module_1.Config()
    var_35 = 'from typing import List  # comment\n'
    var_36 = module_0.file_contents(var_35, var_34)



# Parsed testcases at query #41
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'from sys import exit'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = 'from . import module'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'import os  # noqa'
    var_9 = True
    var_10 = module_1.Config()
    var_11 = module_0.import_type(var_8, var_10)
    assert var_11 is None
    var_12 = 'import os  # NOQA'
    var_13 = module_1.Config()
    var_14 = module_0.import_type(var_12, var_13)
    assert var_14 is None
    var_15 = 'import os  # isort:skip'
    var_16 = module_0.import_type(var_15)
    assert var_16 is None
    var_17 = 'import os  # isort: skip'
    var_18 = module_0.import_type(var_17)
    assert var_18 is None
    var_19 = 'import os  # isort: split'
    var_20 = module_0.import_type(var_19)
    assert var_20 is None
    var_21 = "print('hello')"
    var_22 = module_0.import_type(var_21)
    assert var_22 is None
    var_23 = 'x = 5'
    var_24 = module_0.import_type(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = module_0.import_type(var_8)
    assert var_27 == 'straight'



