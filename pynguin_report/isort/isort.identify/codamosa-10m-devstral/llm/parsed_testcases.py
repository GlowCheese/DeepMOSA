####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = False
    var_6 = '/path/to/file.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 cimport sys'
    var_12 = 15
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = '/another/path.py'
    var_16 = str(var_10)
    assert var_16 == '/another/path.py:15 indented from collections import defaultdict'
    var_17 = 20
    var_18 = 'math'
    var_19 = 'm'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':20 import math as m'



# Parsed testcases at query #2
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import numpy as np'
    var_12 = 3
    var_13 = 'sys'
    var_14 = 'path'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 from sys import path'
    var_17 = 4
    var_18 = True
    var_19 = 'collections'
    var_20 = 'defaultdict'
    var_21 = 'dd'
    var_22 = module_0.Import()
    var_23 = str(var_22)
    assert var_23 == ':4 indented from collections import defaultdict as dd'
    var_24 = 5
    var_25 = 'cython'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 cimport cython'
    var_29 = 6
    var_30 = '/path/to/file.py'
    var_31 = str(var_27)
    assert var_31 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #3
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.imports(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'import numpy as np\n'
    var_8 = [var_7]
    var_9 = iter(var_8)
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'from os import path\n'
    var_14 = [var_13]
    var_15 = iter(var_14)
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'from os import path as p\n'
    var_20 = [var_19]
    var_21 = iter(var_20)
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'cimport numpy as np\n'
    var_26 = [var_25]
    var_27 = iter(var_26)
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'from os import (\n'
    var_32 = '    path,\n'
    var_33 = '    environ\n'
    var_34 = ')\n'
    var_35 = [var_31, var_32, var_33, var_34]
    var_36 = iter(var_35)
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '    import os\n'
    var_41 = [var_40]
    var_42 = iter(var_41)
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '/test/path.py'
    var_47 = [var_0]
    var_48 = iter(var_47)
    var_49 = len(var_44)
    assert var_49 == 1
    var_50 = 'def func():\n'
    var_51 = '    import sys\n'
    var_52 = [var_0, var_50, var_51]
    var_53 = iter(var_52)
    var_54 = True
    var_55 = module_0.imports(var_53, top_only=var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = 'import os  # comment\n'
    var_59 = [var_58]
    var_60 = iter(var_59)
    var_61 = module_0.imports(var_60)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = module_1.Config()
    var_65 = 'import os as os\n'
    var_66 = [var_65]
    var_67 = iter(var_66)
    var_68 = module_0.imports(var_67, var_64)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = '# isort: off'
    var_72 = '# isort: on'
    var_73 = [var_71, var_72]
    var_74 = module_1.Config()
    var_75 = '# isort: off\n'
    var_76 = '# isort: on\n'
    var_77 = [var_75, var_0, var_76, var_1]
    var_78 = iter(var_77)
    var_79 = module_0.imports(var_78, var_74)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 2



# Parsed testcases at query #4
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = True
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import numpy as np'
    var_11 = 3
    var_12 = 'cython'
    var_13 = True
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 cimport cython'
    var_16 = 4
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 from os import path'
    var_20 = 5
    var_21 = True
    var_22 = 'libc'
    var_23 = 'stdio'
    var_24 = 'cstdio'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':5 indented from libc cimport stdio as cstdio'
    var_28 = 6
    var_29 = 'sys'
    var_30 = '/path/to/file.py'
    var_31 = str(var_26)
    assert var_31 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #5
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 2
    var_6 = True
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = var_9.statement()
    assert var_10 == 'import numpy as np'
    var_11 = 3
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'from collections import defaultdict'
    var_16 = 4
    var_17 = True
    var_18 = 'typing'
    var_19 = 'List'
    var_20 = 'TList'
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from typing import List as TList'
    var_23 = 5
    var_24 = 'cython'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'cimport cython'
    var_28 = 6
    var_29 = True
    var_30 = 'libc'
    var_31 = 'lc'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = var_33.statement()
    assert var_34 == 'cimport libc as lc'
    var_35 = 7
    var_36 = 'cdivision'
    var_37 = True
    var_38 = module_0.Import()
    var_39 = var_38.statement()
    assert var_39 == 'from cython cimport cdivision'
    var_40 = 8
    var_41 = True
    var_42 = 'stdio'
    var_43 = 'cstdio'
    var_44 = True
    var_45 = module_0.Import()
    var_46 = var_45.statement()
    assert var_46 == 'from libc cimport stdio as cstdio'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = 2
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 3
    var_8 = 'collections'
    var_9 = 'defaultdict'
    var_10 = 4
    var_11 = 'dd'
    var_12 = 5
    var_13 = 'cython'
    var_14 = True
    var_15 = 6
    var_16 = True
    var_17 = 'sys'
    var_18 = 7
    var_19 = '/path/to/file.py'



# Parsed testcases at query #2
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = False
    var_6 = '/tmp/test.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = 'math'
    var_10 = 'sqrt'
    var_11 = None
    var_12 = module_0.Import()
    var_13 = str(var_12)
    assert var_13 == ':1 from math import sqrt'
    var_14 = 3
    var_15 = 'collections'
    var_16 = 'defaultdict'
    var_17 = str(var_12)
    assert var_17 == '/tmp/test.py:3 indented from collections import defaultdict'
    var_18 = 7
    var_19 = 'numpy'
    var_20 = 'np'
    var_21 = str(var_12)
    assert var_21 == '/tmp/test.py:7 import numpy as np'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = 2
    var_5 = 'sys'
    var_6 = 'from os import path\nfrom sys import argv\n'
    var_7 = 'path'
    var_8 = 'argv'
    var_9 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_10 = 'numpy'
    var_11 = 'np'
    var_12 = 'pandas'
    var_13 = 'DataFrame'
    var_14 = 'df'
    var_15 = 'cimport numpy\nfrom pandas cimport DataFrame\n'
    var_16 = True
    var_17 = True
    var_18 = 'if True:\n    import os\n    from sys import argv\n'
    var_19 = True
    var_20 = 3
    var_21 = True
    var_22 = 'from os import (\n    path,\n    environ,\n)\n'
    var_23 = 'environ'
    var_24 = True
    var_25 = module_0.Config()
    var_26 = 'import numpy as numpy\nfrom pandas import DataFrame as DataFrame\n'
    var_27 = "# This is a comment\nimport os  # inline comment\n'''\nmultiline string\n'''\nimport sys\n"
    var_28 = 5
    var_29 = 'import os; import sys\nfrom pandas import DataFrame; import numpy\n'
    var_30 = '/path/to/file.py'
    var_31 = 'import os\n'
    var_32 = 'import os\ndef foo():\n    import sys\n'
    var_33 = True



# Parsed testcases at query #4
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import numpy as np'
    var_12 = 3
    var_13 = 'sys'
    var_14 = 'path'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 from sys import path'
    var_17 = 4
    var_18 = True
    var_19 = 'collections'
    var_20 = 'defaultdict'
    var_21 = 'dd'
    var_22 = module_0.Import()
    var_23 = str(var_22)
    assert var_23 == ':4 indented from collections import defaultdict as dd'
    var_24 = 5
    var_25 = 'cython'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 cimport cython'
    var_29 = 6
    var_30 = 'json'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import json'



# Parsed testcases at query #5
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'path'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'from os import path'
    var_16 = 'libc'
    var_17 = 'stdio'
    var_18 = True
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from libc cimport stdio'
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = 'dd'
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from collections import defaultdict as dd'



# Parsed testcases at query #6
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from collections import defaultdict\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from pathlib import Path as P\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'cimport numpy\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'from collections import (\n'
    var_27 = '    defaultdict,\n'
    var_28 = '    OrderedDict\n'
    var_29 = ')\n'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = True
    var_35 = module_1.Config()
    var_36 = 'import numpy as numpy\n'
    var_37 = [var_36]
    var_38 = module_0.imports(var_37, var_35)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '    import os\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '/test/path.py'
    var_47 = [var_0]
    var_48 = len(var_44)
    assert var_48 == 1
    var_49 = 'def foo():\n'
    var_50 = '    import sys\n'
    var_51 = [var_0, var_49, var_50]
    var_52 = module_0.imports(var_51, top_only=var_34)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = '# This is a comment\n'
    var_56 = '"""Docstring"""\n'
    var_57 = [var_55, var_0, var_56, var_1]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = 'import os; import sys\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = 'import os \\\n'
    var_67 = '    , sys\n'
    var_68 = [var_66, var_67]
    var_69 = module_0.imports(var_68)
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = 'yield\n'
    var_73 = 'raise Exception\n'
    var_74 = [var_72, var_0, var_73, var_1]
    var_75 = module_0.imports(var_74)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 2



# Parsed testcases at query #7
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import os, sys\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from collections import (\n    defaultdict,\n    OrderedDict\n)\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import os\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = 'import os  # some comment\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'import os; import sys\n'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'def foo():\n'
    var_51 = '    pass\n'
    var_52 = [var_0, var_50, var_51]
    var_53 = True
    var_54 = module_0.imports(var_52, top_only=var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/tmp/test.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = module_1.Config()
    var_61 = 'import numpy as numpy\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62, var_60)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'from typing import (List, Dict)\n'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = 'from typing import \\\n    List, Dict\n'
    var_72 = [var_71]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 2



# Parsed testcases at query #8
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = False
    var_5 = 'os'
    var_6 = module_0.Import()
    var_7 = 2
    var_8 = 'sys'
    var_9 = module_0.Import()
    var_10 = [var_6, var_9]
    var_11 = module_0.imports(var_2)
    var_12 = list(var_11)
    var_13 = 'import numpy as np\n'
    var_14 = 'import pandas as pd\n'
    var_15 = [var_13, var_14]
    var_16 = 'numpy'
    var_17 = 'np'
    var_18 = module_0.Import()
    var_19 = 'pandas'
    var_20 = 'pd'
    var_21 = module_0.Import()
    var_22 = [var_18, var_21]
    var_23 = module_0.imports(var_15)
    var_24 = list(var_23)
    var_25 = 'from collections import defaultdict\n'
    var_26 = 'from typing import List\n'
    var_27 = [var_25, var_26]
    var_28 = 'collections'
    var_29 = 'defaultdict'
    var_30 = module_0.Import()
    var_31 = 'typing'
    var_32 = 'List'
    var_33 = module_0.Import()
    var_34 = [var_30, var_33]
    var_35 = module_0.imports(var_27)
    var_36 = list(var_35)
    var_37 = 'from numpy import array as arr\n'
    var_38 = 'from pandas import DataFrame as DF\n'
    var_39 = [var_37, var_38]
    var_40 = 'array'
    var_41 = 'arr'
    var_42 = module_0.Import()
    var_43 = 'DataFrame'
    var_44 = 'DF'
    var_45 = module_0.Import()
    var_46 = [var_42, var_45]
    var_47 = module_0.imports(var_39)
    var_48 = list(var_47)
    var_49 = 'cimport numpy\n'
    var_50 = 'from numpy cimport ndarray\n'
    var_51 = [var_49, var_50]
    var_52 = True
    var_53 = module_0.Import()
    var_54 = 'ndarray'
    var_55 = True
    var_56 = module_0.Import()
    var_57 = [var_53, var_56]
    var_58 = module_0.imports(var_51)
    var_59 = list(var_58)
    var_60 = '    import os\n'
    var_61 = '    import sys\n'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.Import()
    var_65 = True
    var_66 = module_0.Import()
    var_67 = [var_64, var_66]
    var_68 = module_0.imports(var_62)
    var_69 = list(var_68)
    var_70 = 'from collections import (\n'
    var_71 = '    defaultdict,\n'
    var_72 = '    OrderedDict\n'
    var_73 = ')\n'
    var_74 = [var_70, var_71, var_72, var_73]
    var_75 = module_0.Import()
    var_76 = 'OrderedDict'
    var_77 = module_0.Import()
    var_78 = [var_75, var_77]
    var_79 = module_0.imports(var_74)
    var_80 = list(var_79)
    var_81 = 'import os  # Operating system\n'
    var_82 = 'import sys  # System\n'
    var_83 = [var_81, var_82]
    var_84 = module_0.Import()
    var_85 = module_0.Import()
    var_86 = [var_84, var_85]
    var_87 = module_0.imports(var_83)
    var_88 = list(var_87)
    var_89 = 'import os as os\n'
    var_90 = 'import sys as sys\n'
    var_91 = [var_89, var_90]
    var_92 = module_0.Import()
    var_93 = module_0.Import()
    var_94 = [var_92, var_93]
    var_95 = module_0.imports(var_91)
    var_96 = list(var_95)
    var_97 = [var_0, var_1]
    var_98 = '/path/to/file.py'



# Parsed testcases at query #9
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = False
    var_5 = 'os'
    var_6 = None
    var_7 = 2
    var_8 = 'sys'
    var_9 = module_0.imports(var_2)
    var_10 = list(var_9)
    var_11 = 'import numpy as np\n'
    var_12 = [var_11]
    var_13 = 'numpy'
    var_14 = 'np'
    var_15 = module_0.imports(var_12)
    var_16 = list(var_15)
    var_17 = 'from collections import defaultdict\n'
    var_18 = [var_17]
    var_19 = 'collections'
    var_20 = 'defaultdict'
    var_21 = module_0.imports(var_18)
    var_22 = list(var_21)
    var_23 = 'from pathlib import Path as P\n'
    var_24 = [var_23]
    var_25 = 'pathlib'
    var_26 = 'Path'
    var_27 = 'P'
    var_28 = module_0.imports(var_24)
    var_29 = list(var_28)
    var_30 = 'cimport numpy as np\n'
    var_31 = [var_30]
    var_32 = True
    var_33 = module_0.imports(var_31)
    var_34 = list(var_33)
    var_35 = 'import os, sys\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = '    import os\n'
    var_40 = [var_39]
    var_41 = True
    var_42 = module_0.imports(var_40)
    var_43 = list(var_42)
    var_44 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_45 = [var_44]
    var_46 = 'typing'
    var_47 = 'List'
    var_48 = 3
    var_49 = 'Dict'
    var_50 = module_0.imports(var_45)
    var_51 = list(var_50)
    var_52 = 'x = 1\n'
    var_53 = 'y = 2\n'
    var_54 = [var_52, var_0, var_53]
    var_55 = module_0.imports(var_54)
    var_56 = list(var_55)
    var_57 = '# This is a comment\n'
    var_58 = [var_57, var_0]
    var_59 = module_0.imports(var_58)
    var_60 = list(var_59)
    var_61 = '"""\nThis is a multiline string\n"""\n'
    var_62 = [var_61, var_0]
    var_63 = 4
    var_64 = module_0.imports(var_62)
    var_65 = list(var_64)
    var_66 = 'yield\n'
    var_67 = [var_66, var_0]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = 'raise ValueError\n'
    var_71 = [var_70, var_0]
    var_72 = module_0.imports(var_71)
    var_73 = list(var_72)
    var_74 = 'import os \\\n'
    var_75 = '    import sys\n'
    var_76 = [var_74, var_75]
    var_77 = module_0.imports(var_76)
    var_78 = list(var_77)
    var_79 = '# isort: off\n'
    var_80 = '# isort: on\n'
    var_81 = 'import json\n'
    var_82 = [var_79, var_0, var_1, var_80, var_81]
    var_83 = 5
    var_84 = 'json'
    var_85 = module_0.imports(var_82)
    var_86 = list(var_85)
    var_87 = 'import os as os\n'
    var_88 = [var_87]
    var_89 = []
    var_90 = module_0.imports(var_88)
    var_91 = list(var_90)
    var_92 = 'from os import path as path\n'
    var_93 = [var_92]
    var_94 = []
    var_95 = module_0.imports(var_93)
    var_96 = list(var_95)
    var_97 = module_1.Config()
    var_98 = [var_87]
    var_99 = module_0.imports(var_98, var_97)
    var_100 = list(var_99)
    var_101 = module_1.Config()
    var_102 = [var_92]
    var_103 = 'path'
    var_104 = module_0.imports(var_102, var_101)
    var_105 = list(var_104)
    var_106 = 'def foo():\n'
    var_107 = [var_0, var_106, var_75]
    var_108 = True
    var_109 = module_0.imports(var_107, top_only=var_108)
    var_110 = list(var_109)



# Parsed testcases at query #10
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from collections import defaultdict\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from pathlib import Path as P\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'import os, sys\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'cimport numpy\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = '    import os\n'
    var_32 = [var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = 'from collections import (\n'
    var_37 = '    defaultdict,\n'
    var_38 = '    OrderedDict\n'
    var_39 = ')\n'
    var_40 = [var_36, var_37, var_38, var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'import os  # Operating system interfaces\n'
    var_45 = [var_44]
    var_46 = module_0.imports(var_45)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = 'x = 1\n'
    var_50 = 'y = 2\n'
    var_51 = [var_49, var_0, var_50]
    var_52 = module_0.imports(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'def foo():\n'
    var_56 = '    import sys\n'
    var_57 = [var_0, var_55, var_56]
    var_58 = True
    var_59 = module_0.imports(var_57, top_only=var_58)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 1
    var_62 = [var_0]
    var_63 = '/test.py'
    var_64 = len(var_60)
    assert var_64 == 1
    var_65 = 'import os as os\n'
    var_66 = [var_65]
    var_67 = module_1.Config()
    var_68 = module_0.imports(var_66, var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = []
    var_72 = module_0.imports(var_71)
    var_73 = list(var_72)
    var_74 = len(var_73)
    assert var_74 == 0
    var_75 = 'x = "import os"\n'
    var_76 = [var_75, var_1]
    var_77 = module_0.imports(var_76)
    var_78 = list(var_77)
    var_79 = len(var_78)
    assert var_79 == 1



# Parsed testcases at query #11
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from sys import path\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pandas import DataFrame as df\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import os, sys\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'cimport numpy\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '    import os\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from collections import (\n'
    var_36 = '    OrderedDict,\n'
    var_37 = '    defaultdict\n'
    var_38 = ')\n'
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = '# This is a comment\n'
    var_44 = 'import os  # inline comment\n'
    var_45 = [var_43, var_44]
    var_46 = module_0.imports(var_45)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = '"""docstring"""'
    var_50 = [var_49, var_0]
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = 'def foo():\n'
    var_55 = '    import sys\n'
    var_56 = [var_0, var_54, var_55]
    var_57 = True
    var_58 = module_0.imports(var_56, top_only=var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = '/test/file.py'
    var_62 = [var_0]
    var_63 = len(var_59)
    assert var_63 == 1
    var_64 = module_1.Config()
    var_65 = 'import numpy as numpy\n'
    var_66 = [var_65]
    var_67 = module_0.imports(var_66, var_64)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1
    var_70 = False
    var_71 = 'os'
    var_72 = None
    var_73 = 'path'
    var_74 = 'numpy'
    var_75 = 'np'
    var_76 = 'p'



# Parsed testcases at query #12
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = iter(var_1)
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = iter(var_7)
    var_9 = module_0.imports(var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 'from sys import path\n'
    var_13 = [var_12]
    var_14 = iter(var_13)
    var_15 = module_0.imports(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 'from pandas import DataFrame as df\n'
    var_19 = [var_18]
    var_20 = iter(var_19)
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'cimport numpy\n'
    var_25 = [var_24]
    var_26 = iter(var_25)
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'import os, sys\n'
    var_31 = [var_30]
    var_32 = iter(var_31)
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'from collections import (\n'
    var_37 = '    OrderedDict,\n'
    var_38 = '    defaultdict,\n'
    var_39 = ')\n'
    var_40 = [var_36, var_37, var_38, var_39]
    var_41 = iter(var_40)
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = '    import os\n'
    var_46 = [var_45]
    var_47 = iter(var_46)
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = 'import os  # some comment\n'
    var_52 = [var_51]
    var_53 = iter(var_52)
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = 'import os; import sys\n'
    var_58 = [var_57]
    var_59 = iter(var_58)
    var_60 = module_0.imports(var_59)
    var_61 = list(var_60)
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = '/some/path'
    var_64 = [var_0]
    var_65 = iter(var_64)
    var_66 = len(var_61)
    assert var_66 == 1
    var_67 = 'def foo():\n'
    var_68 = '    import sys\n'
    var_69 = [var_0, var_67, var_68]
    var_70 = iter(var_69)
    var_71 = True
    var_72 = module_0.imports(var_70, top_only=var_71)
    var_73 = list(var_72)
    var_74 = len(var_73)
    assert var_74 == 1
    var_75 = module_1.Config()
    var_76 = 'import numpy as numpy\n'
    var_77 = [var_76]
    var_78 = iter(var_77)
    var_79 = module_0.imports(var_78, var_75)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 1
    var_82 = []
    var_83 = iter(var_82)
    var_84 = module_0.imports(var_83)
    var_85 = list(var_84)
    var_86 = len(var_85)
    assert var_86 == 0
    var_87 = '# comment\n'
    var_88 = '"""docstring"""\n'
    var_89 = 'import sys\n'
    var_90 = [var_87, var_0, var_88, var_89]
    var_91 = iter(var_90)
    var_92 = module_0.imports(var_91)
    var_93 = list(var_92)
    var_94 = len(var_93)
    assert var_94 == 2



# Parsed testcases at query #13
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from sys import path'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'import sys'
    var_16 = [var_0, var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = '    import os'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'import os # comment'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from sys import ('
    var_36 = '    path,'
    var_37 = '    argv'
    var_38 = ')'
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 'from sys import path, \\'
    var_44 = [var_43, var_37]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = []
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 0
    var_52 = 'x = 1'
    var_53 = "print('hello')"
    var_54 = [var_52, var_53]
    var_55 = module_0.imports(var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 0
    var_58 = 'def foo():'
    var_59 = '    import sys'
    var_60 = [var_0, var_58, var_59]
    var_61 = True
    var_62 = module_0.imports(var_60, top_only=var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = module_1.Config()
    var_66 = 'import numpy as numpy'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67, var_65)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #14
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = False
    var_5 = 'os'
    var_6 = 2
    var_7 = 'sys'
    var_8 = module_0.imports(var_2)
    var_9 = list(var_8)
    var_10 = 'from os import path'
    var_11 = 'from sys import argv'
    var_12 = [var_10, var_11]
    var_13 = 'path'
    var_14 = 'argv'
    var_15 = module_0.imports(var_12)
    var_16 = list(var_15)
    var_17 = 'import numpy as np'
    var_18 = 'from pandas import DataFrame as df'
    var_19 = [var_17, var_18]
    var_20 = 'numpy'
    var_21 = 'np'
    var_22 = 'pandas'
    var_23 = 'DataFrame'
    var_24 = 'df'
    var_25 = module_0.imports(var_19)
    var_26 = list(var_25)
    var_27 = 'cimport numpy'
    var_28 = 'from cython cimport int'
    var_29 = [var_27, var_28]
    var_30 = True
    var_31 = 'cython'
    var_32 = 'int'
    var_33 = True
    var_34 = module_0.imports(var_29)
    var_35 = list(var_34)
    var_36 = 'from os import (\n    path,\n    walk\n)'
    var_37 = [var_36]
    var_38 = 'walk'
    var_39 = module_0.imports(var_37)
    var_40 = list(var_39)
    var_41 = '    import os'
    var_42 = '        import sys'
    var_43 = [var_41, var_42]
    var_44 = True
    var_45 = True
    var_46 = module_0.imports(var_43)
    var_47 = list(var_46)
    var_48 = True
    var_49 = module_1.Config()
    var_50 = 'import numpy as numpy'
    var_51 = 'from os import path as path'
    var_52 = [var_50, var_51]
    var_53 = module_0.imports(var_52, var_49)
    var_54 = list(var_53)
    var_55 = '# This is a comment'
    var_56 = "'''Docstring'''"
    var_57 = [var_55, var_0, var_56, var_1]
    var_58 = 4
    var_59 = module_0.imports(var_57)
    var_60 = list(var_59)
    var_61 = 'import os; import sys'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = '/path/to/file.py'
    var_66 = [var_0]



# Parsed testcases at query #15
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import os, sys\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = '    import os\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from collections import (\n'
    var_36 = '    defaultdict,\n'
    var_37 = '    OrderedDict\n'
    var_38 = ')\n'
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = 'import os  # some comment\n'
    var_44 = [var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = 'import os; import sys\n'
    var_49 = [var_48]
    var_50 = module_0.imports(var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = True
    var_54 = module_1.Config()
    var_55 = 'import numpy as numpy\n'
    var_56 = [var_55]
    var_57 = module_0.imports(var_56, var_54)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = '/some/path'
    var_61 = [var_0]
    var_62 = len(var_58)
    assert var_62 == 1
    var_63 = 'def foo():\n'
    var_64 = '    pass\n'
    var_65 = [var_0, var_63, var_64]
    var_66 = module_0.imports(var_65, top_only=var_53)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 1



# Parsed testcases at query #16
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from sys import argv'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'import os, sys'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'from os import (\n    path,\n    sys\n)'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'import os as os'
    var_31 = [var_30]
    var_32 = True
    var_33 = module_1.Config()
    var_34 = module_0.imports(var_31, var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = '    import os'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = 'import os  # This is a comment'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = []
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 0
    var_51 = 'x = 5'
    var_52 = [var_51]
    var_53 = module_0.imports(var_52)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 0
    var_56 = 'from os import \\\n    path'
    var_57 = [var_56]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'numpy'
    var_3 = 'array'
    var_4 = 'np_array'
    var_5 = 'test.py'
    var_6 = 5
    var_7 = False
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 import sys'
    var_12 = 15
    var_13 = 'os.path'
    var_14 = 'join'
    var_15 = 'example.py'
    var_16 = str(var_10)
    assert var_16 == 'example.py:15 from os.path import join'
    var_17 = 20
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = 'script.py'
    var_21 = str(var_10)
    assert var_21 == 'script.py:20 indented from collections import defaultdict'



# Parsed testcases at query #2
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'test_module'
    var_3 = 'test_attribute'
    var_4 = 'test_alias'
    var_5 = '/test/path'
    var_6 = 5
    var_7 = False
    var_8 = None
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':5 import test_attribute from test_module'
    var_11 = 3
    var_12 = '/another/path'
    var_13 = str(var_9)
    assert var_13 == '/another/path:3 indented cimport test_module as test_alias'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':1 import test_module'



# Parsed testcases at query #3
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = False
    var_5 = 'os'
    var_6 = 2
    var_7 = 'sys'
    var_8 = module_0.imports(var_2)
    var_9 = list(var_8)
    var_10 = 'import numpy as np\n'
    var_11 = 'import pandas as pd\n'
    var_12 = [var_10, var_11]
    var_13 = 'numpy'
    var_14 = 'np'
    var_15 = 'pandas'
    var_16 = 'pd'
    var_17 = module_0.imports(var_12)
    var_18 = list(var_17)
    var_19 = 'from collections import defaultdict\n'
    var_20 = 'from typing import List\n'
    var_21 = [var_19, var_20]
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = 'typing'
    var_25 = 'List'
    var_26 = module_0.imports(var_21)
    var_27 = list(var_26)
    var_28 = 'from numpy import array as arr\n'
    var_29 = 'from pandas import DataFrame as df\n'
    var_30 = [var_28, var_29]
    var_31 = 'array'
    var_32 = 'arr'
    var_33 = 'DataFrame'
    var_34 = 'df'
    var_35 = module_0.imports(var_30)
    var_36 = list(var_35)
    var_37 = 'cimport numpy as np\n'
    var_38 = 'from numpy cimport ndarray\n'
    var_39 = [var_37, var_38]
    var_40 = True
    var_41 = 'ndarray'
    var_42 = True
    var_43 = module_0.imports(var_39)
    var_44 = list(var_43)
    var_45 = 'def foo():\n'
    var_46 = '    import os\n'
    var_47 = '    import sys\n'
    var_48 = [var_45, var_46, var_47]
    var_49 = True
    var_50 = 3
    var_51 = True
    var_52 = module_0.imports(var_48)
    var_53 = list(var_52)
    var_54 = 'from collections import (\n'
    var_55 = '    defaultdict,\n'
    var_56 = '    OrderedDict\n'
    var_57 = ')\n'
    var_58 = [var_54, var_55, var_56, var_57]
    var_59 = 'OrderedDict'
    var_60 = module_0.imports(var_58)
    var_61 = list(var_60)
    var_62 = 'import os  # Operating system\n'
    var_63 = 'import sys  # System\n'
    var_64 = [var_62, var_63]
    var_65 = module_0.imports(var_64)
    var_66 = list(var_65)
    var_67 = 'x = 1\n'
    var_68 = 'y = 2\n'
    var_69 = [var_67, var_0, var_68, var_1]
    var_70 = 4
    var_71 = module_0.imports(var_69)
    var_72 = list(var_71)
    var_73 = [var_0, var_45, var_47]
    var_74 = True
    var_75 = module_0.imports(var_73, top_only=var_74)
    var_76 = list(var_75)



# Parsed testcases at query #4
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = 'test.py'
    var_11 = 3
    var_12 = 'sys'
    var_13 = 'path'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from sys import path'
    var_16 = 4
    var_17 = 'cython'
    var_18 = 'cy'
    var_19 = True
    var_20 = 'module.pyx'
    var_21 = 5
    var_22 = True
    var_23 = 'libc'
    var_24 = 'stdio'
    var_25 = 'cstdio'
    var_26 = True
    var_27 = 'example.pyx'



# Parsed testcases at query #5
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'cy'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'p'
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from os import path as p'
    var_23 = 'view'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport view'
    var_27 = 'v'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport view as v'



# Parsed testcases at query #6
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = '/test.py'
    var_6 = 2
    var_7 = True
    var_8 = 'sys'
    var_9 = None
    var_10 = True
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':2 indented cimport sys'



# Parsed testcases at query #7
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from collections import defaultdict\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from pathlib import Path as P\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'cimport numpy\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'from collections import (\n'
    var_27 = '    defaultdict,\n'
    var_28 = '    OrderedDict,\n'
    var_29 = ')\n'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = True
    var_35 = module_1.Config()
    var_36 = 'import numpy as numpy\n'
    var_37 = [var_36]
    var_38 = module_0.imports(var_37, var_35)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '    import os\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '/path/to/file.py'
    var_47 = [var_1]
    var_48 = len(var_44)
    assert var_48 == 1
    var_49 = 'def foo():\n'
    var_50 = '    import sys\n'
    var_51 = [var_0, var_49, var_50]
    var_52 = module_0.imports(var_51, top_only=var_34)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'import os  # some comment\n'
    var_56 = [var_55]
    var_57 = module_0.imports(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = 'import os; import sys\n'
    var_61 = [var_60]
    var_62 = module_0.imports(var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = 'from collections import \\\n'
    var_66 = '    defaultdict\n'
    var_67 = [var_65, var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #8
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == ':2 import numpy as np'
    var_10 = 3
    var_11 = 'libc'
    var_12 = True
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 cimport libc'
    var_15 = 4
    var_16 = 'path'
    var_17 = module_0.Import()
    var_18 = str(var_17)
    assert var_18 == ':4 from os import path'
    var_19 = 5
    var_20 = 'osp'
    var_21 = module_0.Import()
    var_22 = str(var_21)
    assert var_22 == ':5 from os import path as osp'
    var_23 = 6
    var_24 = True
    var_25 = 'sys'
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':6 indented import sys'
    var_28 = 7
    var_29 = '/path/to/file.py'
    var_30 = 8
    var_31 = 'stdio'
    var_32 = 'cstdio'
    var_33 = True
    var_34 = module_0.Import()
    var_35 = str(var_34)
    assert var_35 == ':8 from libc cimport stdio as cstdio'



# Parsed testcases at query #9
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = True
    var_7 = 'sys'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == ':2 indented import sys'
    var_10 = 3
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 from collections import defaultdict'
    var_15 = 4
    var_16 = 'numpy'
    var_17 = 'np'
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 import numpy as np'
    var_20 = 5
    var_21 = True
    var_22 = 'pandas'
    var_23 = 'DataFrame'
    var_24 = 'pd'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 indented from pandas import DataFrame as pd'
    var_27 = 6
    var_28 = 'cython'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':6 cimport cython'
    var_32 = 7
    var_33 = 'pathlib'
    var_34 = '/tmp/test.py'
    var_35 = str(var_30)
    assert var_35 == '/tmp/test.py:7 import pathlib'
    var_36 = 8
    var_37 = True
    var_38 = 'django'
    var_39 = 'models'
    var_40 = 'dm'
    var_41 = True
    var_42 = '/home/user/project/main.py'
    var_43 = str(var_30)
    assert var_43 == '/home/user/project/main.py:8 indented from django cimport models as dm'



# Parsed testcases at query #10
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'sys'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import sys'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'os'
    var_10 = 'path'
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'from os import path'
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = 'dd'
    var_16 = module_0.Import()
    var_17 = var_16.statement()
    assert var_17 == 'from collections import defaultdict as dd'
    var_18 = 'cython'
    var_19 = True
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'cimport cython'
    var_22 = 'cy'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'cimport cython as cy'
    var_26 = 'view'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport view'
    var_30 = 'v'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = var_32.statement()
    assert var_33 == 'from cython cimport view as v'



# Parsed testcases at query #11
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'sys'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import sys'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'os'
    var_10 = 'path'
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'from os import path'
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = 'dd'
    var_16 = module_0.Import()
    var_17 = var_16.statement()
    assert var_17 == 'from collections import defaultdict as dd'
    var_18 = 'cython'
    var_19 = True
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'cimport cython'
    var_22 = 'cy'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'cimport cython as cy'
    var_26 = 'view'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport view'
    var_30 = 'cv'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = var_32.statement()
    assert var_33 == 'from cython cimport view as cv'



# Parsed testcases at query #12
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import os'
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = var_8.statement()
    assert var_9 == 'import numpy as np'
    var_10 = 'cython'
    var_11 = True
    var_12 = module_0.Import()
    var_13 = var_12.statement()
    assert var_13 == 'cimport cython'
    var_14 = 'cy'
    var_15 = True
    var_16 = module_0.Import()
    var_17 = var_16.statement()
    assert var_17 == 'cimport cython as cy'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'p'
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from os import path as p'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport cython'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport cython as cy'



# Parsed testcases at query #13
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from sys import path\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from collections import OrderedDict as OD\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import os, sys\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'cimport numpy\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = True
    var_36 = module_1.Config()
    var_37 = 'import numpy as numpy\n'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38, var_36)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = '    import os\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = '/test/file.py'
    var_48 = [var_0]
    var_49 = len(var_45)
    assert var_49 == 1
    var_50 = 'def func():\n'
    var_51 = '    pass\n'
    var_52 = [var_0, var_50, var_51]
    var_53 = module_0.imports(var_52, top_only=var_35)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1
    var_56 = '# This is a comment\n'
    var_57 = [var_56, var_0]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = 'import os; import sys\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = 'from typing import \\\n    List\n'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #14
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import sys, os\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import pandas\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = 'import math  # math module\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'import json; import ast\n'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'import re\n'
    var_51 = 'def foo():\n'
    var_52 = '    import bar\n'
    var_53 = [var_50, var_51, var_52]
    var_54 = True
    var_55 = module_0.imports(var_53, top_only=var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = module_1.Config()
    var_59 = 'import pandas as pandas\n'
    var_60 = [var_59]
    var_61 = module_0.imports(var_60, var_58)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = '/test/file.py'
    var_65 = 'import sys\n'
    var_66 = [var_65]
    var_67 = len(var_62)
    assert var_67 == 1



# Parsed testcases at query #15
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import numpy as np'
    var_12 = 3
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 from collections import defaultdict'
    var_17 = 4
    var_18 = True
    var_19 = 'typing'
    var_20 = 'List'
    var_21 = 'list'
    var_22 = module_0.Import()
    var_23 = str(var_22)
    assert var_23 == ':4 indented from typing import List as list'
    var_24 = 5
    var_25 = 'cython'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 cimport cython'
    var_29 = 6
    var_30 = 'sys'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #16
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 1
    var_7 = False
    var_8 = 'os'
    var_9 = 2
    var_10 = 'sys'
    var_11 = 'import numpy as np\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'numpy'
    var_17 = 'np'
    var_18 = 'from collections import defaultdict\n'
    var_19 = [var_18]
    var_20 = module_0.imports(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'collections'
    var_24 = 'defaultdict'
    var_25 = 'from pathlib import Path as P\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'pathlib'
    var_31 = 'Path'
    var_32 = 'P'
    var_33 = 'cimport numpy\n'
    var_34 = [var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 1
    var_38 = True
    var_39 = 'import os, sys\n'
    var_40 = [var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = '    import os\n'
    var_45 = [var_44]
    var_46 = module_0.imports(var_45)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = True
    var_50 = 'from collections import (\n    defaultdict,\n    OrderedDict\n)\n'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = 'OrderedDict'
    var_56 = 'from collections import defaultdict, \\\n    OrderedDict\n'
    var_57 = [var_56]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = '# This is a comment\nimport os\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = '# isort: off'
    var_67 = '# isort: on'
    var_68 = [var_66, var_67]
    var_69 = module_1.Config()
    var_70 = '# isort: off\nimport os\n# isort: on\nimport sys\n'
    var_71 = [var_70]
    var_72 = module_0.imports(var_71, var_69)
    var_73 = list(var_72)
    var_74 = len(var_73)
    assert var_74 == 2
    var_75 = 4
    var_76 = 'import os\ndef foo():\n    pass\n'
    var_77 = [var_76]
    var_78 = True
    var_79 = module_0.imports(var_77, top_only=var_78)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 1
    var_82 = '/path/to/file.py'
    var_83 = [var_0]
    var_84 = len(var_80)
    assert var_84 == 1
    var_85 = True
    var_86 = module_1.Config()
    var_87 = 'import numpy as numpy\n'
    var_88 = [var_87]
    var_89 = module_0.imports(var_88, var_86)
    var_90 = list(var_89)
    var_91 = len(var_90)
    assert var_91 == 1
    var_92 = []
    var_93 = module_0.imports(var_92)
    var_94 = list(var_93)
    var_95 = len(var_94)
    assert var_95 == 0



# Parsed testcases at query #17
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'cy'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'libc'
    var_21 = 'stdio'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from libc cimport stdio'
    var_25 = 'p'
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from os import path as p'
    var_28 = 's'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from libc cimport stdio as s'



# Parsed testcases at query #18
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = False
    var_6 = '/test.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 cimport sys'
    var_12 = 15
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = '/example.py'
    var_16 = str(var_10)
    assert var_16 == '/example.py:15 indented from collections import defaultdict'
    var_17 = 20
    var_18 = 'numpy'
    var_19 = 'np'
    var_20 = '/script.py'
    var_21 = str(var_10)
    assert var_21 == '/script.py:20 import numpy as np'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = 'numpy'
    var_5 = True
    var_6 = 'np'
    var_7 = True
    var_8 = 'path'
    var_9 = 'p'
    var_10 = 'array'
    var_11 = True
    var_12 = 'arr'
    var_13 = True



# Parsed testcases at query #20
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = 1
    var_3 = False
    var_4 = 'os'
    var_5 = module_0.imports(var_1)
    var_6 = list(var_5)
    var_7 = 'import numpy as np'
    var_8 = [var_7]
    var_9 = 'numpy'
    var_10 = 'np'
    var_11 = module_0.imports(var_8)
    var_12 = list(var_11)
    var_13 = 'from sys import argv'
    var_14 = [var_13]
    var_15 = 'sys'
    var_16 = 'argv'
    var_17 = module_0.imports(var_14)
    var_18 = list(var_17)
    var_19 = 'cimport numpy'
    var_20 = [var_19]
    var_21 = True
    var_22 = module_0.imports(var_20)
    var_23 = list(var_22)
    var_24 = 'import os, sys'
    var_25 = [var_24]
    var_26 = module_0.imports(var_25)
    var_27 = list(var_26)
    var_28 = '    import os'
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.imports(var_29)
    var_32 = list(var_31)
    var_33 = 'import os  # comment'
    var_34 = [var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = 'from collections import (OrderedDict,\n    defaultdict)'
    var_38 = [var_37]
    var_39 = 'collections'
    var_40 = 'OrderedDict'
    var_41 = 2
    var_42 = 'defaultdict'
    var_43 = module_0.imports(var_38)
    var_44 = list(var_43)
    var_45 = True
    var_46 = module_1.Config()
    var_47 = 'import numpy as numpy'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48, var_46)
    var_50 = list(var_49)
    var_51 = '/test.py'
    var_52 = [var_0]



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = 2
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = 3
    var_8 = 'module'
    var_9 = True
    var_10 = 4
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 5
    var_14 = 'libc'
    var_15 = 'stdio'
    var_16 = 'c_stdio'
    var_17 = True
    var_18 = 6
    var_19 = True
    var_20 = 'sys'
    var_21 = 7
    var_22 = 'pathlib'
    var_23 = '/some/path.py'



# Parsed testcases at query #22
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = True
    var_7 = 'sys'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == ':2 indented import sys'
    var_10 = 3
    var_11 = 'path'
    var_12 = module_0.Import()
    var_13 = str(var_12)
    assert var_13 == ':3 from os import path'
    var_14 = 4
    var_15 = 'numpy'
    var_16 = 'np'
    var_17 = module_0.Import()
    var_18 = str(var_17)
    assert var_18 == ':4 import numpy as np'
    var_19 = 5
    var_20 = 'pandas'
    var_21 = 'DataFrame'
    var_22 = 'pd'
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':5 from pandas import DataFrame as pd'
    var_25 = 6
    var_26 = 'libc'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 cimport libc'
    var_30 = 7
    var_31 = '/path/to/file.py'
    var_32 = str(var_28)
    assert var_32 == '/path/to/file.py:7 import sys'
    var_33 = 8
    var_34 = True
    var_35 = 'tensorflow'
    var_36 = 'keras'
    var_37 = 'tf'
    var_38 = True
    var_39 = str(var_28)
    assert var_39 == '/path/to/file.py:8 indented from tensorflow cimport keras as tf'



# Parsed testcases at query #23
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import numpy as np'
    var_12 = 3
    var_13 = 'cython'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 cimport cython'
    var_17 = 4
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from os import path'
    var_21 = 5
    var_22 = True
    var_23 = 'libc'
    var_24 = 'stdio'
    var_25 = 'libc_stdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 indented from libc cimport stdio as libc_stdio'
    var_29 = 6
    var_30 = 'sys'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #24
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import sys, os'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'cimport numpy'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'from typing import ('
    var_31 = '    List,'
    var_32 = [var_30, var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = '    import sys'
    var_37 = [var_36]
    var_38 = module_0.imports(var_37)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '/test/file.py'
    var_42 = [var_0]
    var_43 = len(var_39)
    assert var_43 == 1
    var_44 = True
    var_45 = module_1.Config()
    var_46 = 'import os as os'
    var_47 = [var_46]
    var_48 = module_0.imports(var_47, var_45)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = 'def foo():'
    var_52 = [var_0, var_51, var_36]
    var_53 = module_0.imports(var_52, top_only=var_44)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1



# Parsed testcases at query #25
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import os'
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = var_8.statement()
    assert var_9 == 'import numpy as np'
    var_10 = 'cython'
    var_11 = True
    var_12 = module_0.Import()
    var_13 = var_12.statement()
    assert var_13 == 'cimport cython'
    var_14 = 'cy'
    var_15 = True
    var_16 = module_0.Import()
    var_17 = var_16.statement()
    assert var_17 == 'cimport cython as cy'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport cython'
    var_24 = 'p'
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport cython as cy'



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = 2
    var_5 = 'sys'
    var_6 = 'from os import path\nfrom sys import argv\n'
    var_7 = 'path'
    var_8 = 'argv'
    var_9 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_10 = 'numpy'
    var_11 = 'np'
    var_12 = 'pandas'
    var_13 = 'DataFrame'
    var_14 = 'df'
    var_15 = 'cimport numpy\nfrom pandas cimport DataFrame\n'
    var_16 = True
    var_17 = True
    var_18 = 'def foo():\n    import os\n    from sys import argv\n'
    var_19 = True
    var_20 = 3
    var_21 = True
    var_22 = 'from os import (\n    path,\n    environ,\n)\n'
    var_23 = 'environ'
    var_24 = 'import os  # comment\n# comment\nimport sys\n'
    var_25 = 'import os; import sys\n'
    var_26 = True
    var_27 = module_0.Config()
    var_28 = 'import numpy as numpy\nfrom pandas import DataFrame as DataFrame\n'



# Parsed testcases at query #27
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import sys, os\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'cimport numpy\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '    import os\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from collections import (\n    defaultdict,\n    Counter\n)\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = True
    var_41 = module_1.Config()
    var_42 = 'import numpy as numpy\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43, var_41)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 'import os  # comment\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'def foo():\n'
    var_53 = '    import sys\n'
    var_54 = [var_0, var_52, var_53]
    var_55 = module_0.imports(var_54, top_only=var_40)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = '/test/path.py'
    var_59 = [var_0]
    var_60 = len(var_56)
    assert var_60 == 1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.imports(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'import numpy as np\n'
    var_8 = [var_7]
    var_9 = iter(var_8)
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 'from collections import defaultdict\n'
    var_14 = [var_13]
    var_15 = iter(var_14)
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'from pathlib import Path as P\n'
    var_20 = [var_19]
    var_21 = iter(var_20)
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'cimport numpy\n'
    var_26 = [var_25]
    var_27 = iter(var_26)
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'from collections import (\n'
    var_32 = '    defaultdict,\n'
    var_33 = '    Counter\n'
    var_34 = ')\n'
    var_35 = [var_31, var_32, var_33, var_34]
    var_36 = iter(var_35)
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = True
    var_41 = module_1.Config()
    var_42 = 'import numpy as numpy\n'
    var_43 = [var_42]
    var_44 = iter(var_43)
    var_45 = module_0.imports(var_44, var_41)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = '    import os\n'
    var_49 = [var_48]
    var_50 = iter(var_49)
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = '/test/path.py'
    var_55 = [var_0]
    var_56 = iter(var_55)
    var_57 = len(var_52)
    assert var_57 == 1
    var_58 = 'def foo():\n'
    var_59 = '    import sys\n'
    var_60 = [var_0, var_58, var_59]
    var_61 = iter(var_60)
    var_62 = module_0.imports(var_61, top_only=var_40)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = '# comment\n'
    var_66 = '"""docstring"""\n'
    var_67 = [var_65, var_0, var_66, var_1]
    var_68 = iter(var_67)
    var_69 = module_0.imports(var_68)
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 2



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = 2
    var_5 = 'sys'
    var_6 = 'from os import path\nfrom sys import argv\n'
    var_7 = 'path'
    var_8 = 'argv'
    var_9 = 'cimport numpy\nfrom numpy cimport ndarray\n'
    var_10 = 'numpy'
    var_11 = True
    var_12 = 'ndarray'
    var_13 = True
    var_14 = 'import numpy as np\nfrom os import path as p\n'
    var_15 = 'np'
    var_16 = 'p'
    var_17 = 'import numpy as numpy\nfrom os import path as path\n'
    var_18 = True
    var_19 = module_0.Config()
    var_20 = 'from os import (\n    path,\n    sep\n)\n'
    var_21 = 3
    var_22 = 'sep'
    var_23 = 'if True:\n    import sys\n'
    var_24 = True
    var_25 = 'import os  # comment\nfrom sys import argv  # another comment\n'
    var_26 = 'import os; import sys\n'
    var_27 = 'import os\ndef foo():\n    import sys\n'
    var_28 = True



# Parsed testcases at query #3
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = '/test/path.py'
    var_6 = 2
    var_7 = True
    var_8 = 'sys'
    var_9 = None
    var_10 = True
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':2 indented cimport sys'
    var_13 = 3
    var_14 = 'numpy'
    var_15 = 'array'
    var_16 = 'script.py'
    var_17 = str(var_11)
    assert var_17 == 'script.py:3 import numpy.array'



# Parsed testcases at query #4
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = 'import sys'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = False
    var_5 = 'os'
    var_6 = 2
    var_7 = 'sys'
    var_8 = module_0.imports(var_2)
    var_9 = list(var_8)
    var_10 = 'from os import path'
    var_11 = 'from sys import argv'
    var_12 = [var_10, var_11]
    var_13 = 'path'
    var_14 = 'argv'
    var_15 = module_0.imports(var_12)
    var_16 = list(var_15)
    var_17 = 'import numpy as np'
    var_18 = 'from pandas import DataFrame as df'
    var_19 = [var_17, var_18]
    var_20 = 'numpy'
    var_21 = 'np'
    var_22 = 'pandas'
    var_23 = 'DataFrame'
    var_24 = 'df'
    var_25 = module_0.imports(var_19)
    var_26 = list(var_25)
    var_27 = 'cimport numpy'
    var_28 = 'from cython cimport int'
    var_29 = [var_27, var_28]
    var_30 = True
    var_31 = 'cython'
    var_32 = 'int'
    var_33 = True
    var_34 = module_0.imports(var_29)
    var_35 = list(var_34)
    var_36 = 'from os import ('
    var_37 = '    path,'
    var_38 = '    environ'
    var_39 = ')'
    var_40 = [var_36, var_37, var_38, var_39]
    var_41 = 'environ'
    var_42 = module_0.imports(var_40)
    var_43 = list(var_42)
    var_44 = '    import os'
    var_45 = '        import sys'
    var_46 = [var_44, var_45]
    var_47 = True
    var_48 = True
    var_49 = module_0.imports(var_46)
    var_50 = list(var_49)
    var_51 = '# This is a comment'
    var_52 = 'import os  # inline comment'
    var_53 = "'''docstring'''"
    var_54 = [var_51, var_52, var_53, var_1]
    var_55 = 4
    var_56 = module_0.imports(var_54)
    var_57 = list(var_56)
    var_58 = 'import os as os'
    var_59 = 'from sys import argv as argv'
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_1.Config()
    var_63 = module_0.imports(var_60, var_62)
    var_64 = list(var_63)
    var_65 = [var_0]
    var_66 = '/path/to/file.py'
    var_67 = 'def foo():'
    var_68 = '    import sys'
    var_69 = [var_0, var_67, var_68]
    var_70 = True
    var_71 = module_0.imports(var_69, top_only=var_70)
    var_72 = list(var_71)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'module'
    var_3 = None
    var_4 = 'alias'
    var_5 = True
    var_6 = True
    var_7 = 'attribute'
    var_8 = True
    var_9 = True



# Parsed testcases at query #6
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'sys'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import sys'
    var_5 = 5
    var_6 = True
    var_7 = 'os'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == ':5 indented import os'
    var_10 = 10
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':10 import numpy as np'
    var_15 = 15
    var_16 = True
    var_17 = 'pandas'
    var_18 = 'pd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':15 indented import pandas as pd'
    var_21 = 20
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':20 from collections import defaultdict'
    var_26 = 25
    var_27 = 'typing'
    var_28 = 'List'
    var_29 = 'TList'
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':25 from typing import List as TList'
    var_32 = 30
    var_33 = 'cython'
    var_34 = True
    var_35 = module_0.Import()
    var_36 = str(var_35)
    assert var_36 == ':30 cimport cython'
    var_37 = 35
    var_38 = 'cy'
    var_39 = True
    var_40 = module_0.Import()
    var_41 = str(var_40)
    assert var_41 == ':35 cimport cython as cy'
    var_42 = 40
    var_43 = '/path/to/file.py'
    var_44 = str(var_40)
    assert var_44 == '/path/to/file.py:40 import sys'



# Parsed testcases at query #7
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = True
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import numpy as np'
    var_11 = 3
    var_12 = 'sys'
    var_13 = 'path'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from sys import path'
    var_16 = 4
    var_17 = True
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = 'dd'
    var_21 = module_0.Import()
    var_22 = str(var_21)
    assert var_22 == ':4 indented from collections import defaultdict as dd'
    var_23 = 5
    var_24 = 'cython'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':5 cimport cython'
    var_28 = 6
    var_29 = '/path/to/file.py'
    var_30 = str(var_26)
    assert var_30 == '/path/to/file.py:6 import os'



# Parsed testcases at query #8
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'cy'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'p'
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from os import path as p'
    var_23 = 'view'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport view'
    var_27 = 'v'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport view as v'



# Parsed testcases at query #9
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == ':1 import os'
    var_5 = 2
    var_6 = True
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import numpy as np'
    var_11 = 3
    var_12 = 'cython'
    var_13 = True
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 cimport cython'
    var_16 = 4
    var_17 = 'sys'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from sys import path'
    var_21 = 5
    var_22 = True
    var_23 = 'collections'
    var_24 = 'defaultdict'
    var_25 = 'dd'
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':5 indented from collections import defaultdict as dd'
    var_28 = 6
    var_29 = 'json'
    var_30 = '/path/to/file.py'
    var_31 = str(var_26)
    assert var_31 == '/path/to/file.py:6 import json'



# Parsed testcases at query #10
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'cy'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'cfunc'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport cfunc'
    var_24 = 'p'
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = 'cf'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #11
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from os import path\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from os import path as p\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'cimport numpy\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'from os import (\n'
    var_27 = '    path,\n'
    var_28 = '    sys\n'
    var_29 = ')\n'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = True
    var_35 = module_1.Config()
    var_36 = 'import numpy as numpy\n'
    var_37 = [var_36]
    var_38 = module_0.imports(var_37, var_35)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '    import os\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = 'def foo():\n'
    var_47 = '    import sys\n'
    var_48 = [var_0, var_46, var_47]
    var_49 = module_0.imports(var_48, top_only=var_34)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = '/test/path'
    var_53 = [var_0]
    var_54 = len(var_50)
    assert var_54 == 1



# Parsed testcases at query #12
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = 'cy'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'libc'
    var_21 = 'stdio'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from libc cimport stdio'
    var_25 = 'p'
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from os import path as p'
    var_28 = 's'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from libc cimport stdio as s'



# Parsed testcases at query #13
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = 'test.py'
    var_11 = str(var_4)
    assert var_11 == 'test.py:2 indented import numpy as np'
    var_12 = 3
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 from collections import defaultdict'
    var_17 = 4
    var_18 = True
    var_19 = 'typing'
    var_20 = 'List'
    var_21 = 'TList'
    var_22 = 'example.py'
    var_23 = str(var_15)
    assert var_23 == 'example.py:4 indented from typing import List as TList'
    var_24 = 5
    var_25 = 'cython'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 cimport cython'
    var_29 = 6
    var_30 = True
    var_31 = 'libc'
    var_32 = 'stdio'
    var_33 = True
    var_34 = 'cy.py'
    var_35 = str(var_27)
    assert var_35 == 'cy.py:6 indented from libc cimport stdio'



# Parsed testcases at query #14
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = False
    var_6 = '/test/path.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 cimport sys'
    var_12 = 15
    var_13 = 'collections'
    var_14 = 'coll'
    var_15 = 'example.py'
    var_16 = str(var_10)
    assert var_16 == 'example.py:15 indented import collections as coll'
    var_17 = 20
    var_18 = 'math'
    var_19 = 'sqrt'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':20 from math cimport sqrt'



# Parsed testcases at query #15
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = 1
    var_3 = False
    var_4 = 'os'
    var_5 = module_0.imports(var_1)
    var_6 = list(var_5)
    var_7 = 'import numpy as np'
    var_8 = [var_7]
    var_9 = 'numpy'
    var_10 = 'np'
    var_11 = module_0.imports(var_8)
    var_12 = list(var_11)
    var_13 = 'from sys import argv'
    var_14 = [var_13]
    var_15 = 'sys'
    var_16 = 'argv'
    var_17 = module_0.imports(var_14)
    var_18 = list(var_17)
    var_19 = 'cimport numpy'
    var_20 = [var_19]
    var_21 = True
    var_22 = module_0.imports(var_20)
    var_23 = list(var_22)
    var_24 = 'import os, sys'
    var_25 = [var_24]
    var_26 = module_0.imports(var_25)
    var_27 = list(var_26)
    var_28 = '    import os'
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.imports(var_29)
    var_32 = list(var_31)
    var_33 = 'from os import (\n    path,\n    sep\n)'
    var_34 = [var_33]
    var_35 = 'path'
    var_36 = 'sep'
    var_37 = module_0.imports(var_34)
    var_38 = list(var_37)
    var_39 = 'from os import path, \\\n    sep'
    var_40 = [var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = True
    var_44 = module_1.Config()
    var_45 = 'import numpy as numpy'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46, var_44)
    var_48 = list(var_47)
    var_49 = 'import os # This is a comment'
    var_50 = [var_49]
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = []
    var_54 = []
    var_55 = module_0.imports(var_53)
    var_56 = list(var_55)
    var_57 = 'x = 1'
    var_58 = [var_57]
    var_59 = []
    var_60 = module_0.imports(var_58)
    var_61 = list(var_60)
    var_62 = 'def foo():'
    var_63 = '    import sys'
    var_64 = [var_0, var_62, var_63]
    var_65 = True
    var_66 = module_0.imports(var_64, top_only=var_65)
    var_67 = list(var_66)



# Parsed testcases at query #16
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 1
    var_7 = False
    var_8 = 'os'
    var_9 = None
    var_10 = 2
    var_11 = 'sys'
    var_12 = 'from os import path\n'
    var_13 = 'from sys import argv\n'
    var_14 = [var_12, var_13]
    var_15 = module_0.imports(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'path'
    var_19 = 'argv'
    var_20 = 'import numpy as np\n'
    var_21 = 'import pandas as pd\n'
    var_22 = [var_20, var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'numpy'
    var_27 = 'np'
    var_28 = 'pandas'
    var_29 = 'pd'
    var_30 = 'from os import path as p\n'
    var_31 = 'from sys import argv as a\n'
    var_32 = [var_30, var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'p'
    var_37 = 'a'
    var_38 = 'cimport numpy as np\n'
    var_39 = 'cimport pandas as pd\n'
    var_40 = [var_38, var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = True
    var_45 = True
    var_46 = 'from os import (\n'
    var_47 = '    path,\n'
    var_48 = '    environ\n'
    var_49 = ')\n'
    var_50 = [var_46, var_47, var_48, var_49]
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = 3
    var_55 = 'environ'
    var_56 = '    import os\n'
    var_57 = '    import sys\n'
    var_58 = [var_56, var_57]
    var_59 = module_0.imports(var_58)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = True
    var_63 = True
    var_64 = [var_0, var_1]
    var_65 = '/test/path'
    var_66 = len(var_60)
    assert var_66 == 2
    var_67 = 'def func():\n'
    var_68 = [var_0, var_67, var_57]
    var_69 = True
    var_70 = module_0.imports(var_68, top_only=var_69)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 1
    var_73 = True
    var_74 = module_1.Config()
    var_75 = 'import os as os\n'
    var_76 = 'from sys import path as path\n'
    var_77 = [var_75, var_76]
    var_78 = module_0.imports(var_77, var_74)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = 'import os  # comment\n'
    var_82 = 'import sys  # another comment\n'
    var_83 = [var_81, var_82]
    var_84 = module_0.imports(var_83)
    var_85 = list(var_84)
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = 'import os; import sys\n'
    var_88 = [var_87]
    var_89 = module_0.imports(var_88)
    var_90 = list(var_89)
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = 'import os \\\n'
    var_93 = '    , sys\n'
    var_94 = [var_92, var_93]
    var_95 = module_0.imports(var_94)
    var_96 = list(var_95)
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = 'from os import (path, environ)\n'
    var_99 = [var_98]
    var_100 = module_0.imports(var_99)
    var_101 = list(var_100)
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = 'from os cimport path\n'
    var_104 = [var_103]
    var_105 = module_0.imports(var_104)
    var_106 = list(var_105)
    var_107 = len(var_106)
    assert var_107 == 1
    var_108 = True
    var_109 = 'yield\n'
    var_110 = [var_109, var_0]
    var_111 = module_0.imports(var_110)
    var_112 = list(var_111)
    var_113 = len(var_112)
    assert var_113 == 1
    var_114 = 'raise\n'
    var_115 = [var_114, var_0]
    var_116 = module_0.imports(var_115)
    var_117 = list(var_116)
    var_118 = len(var_117)
    assert var_118 == 1
    var_119 = '\n'
    var_120 = [var_119, var_0, var_119, var_1]
    var_121 = module_0.imports(var_120)
    var_122 = list(var_121)
    var_123 = len(var_122)
    assert var_123 == 2
    var_124 = 4



# Parsed testcases at query #17
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = False
    var_6 = '/tmp/test.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 import sys'
    var_12 = 15
    var_13 = 'libc'
    var_14 = 'stdio'
    var_15 = 'test.py'
    var_16 = str(var_10)
    assert var_16 == 'test.py:15 indented from libc cimport stdio'
    var_17 = 20
    var_18 = 'collections'
    var_19 = 'coll'
    var_20 = '/path/to/file.py'
    var_21 = str(var_10)
    assert var_21 == '/path/to/file.py:20 import collections as coll'



# Parsed testcases at query #18
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import numpy as np'
    var_12 = 3
    var_13 = 'libc'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 cimport libc'
    var_17 = 4
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 from collections import defaultdict'
    var_22 = 5
    var_23 = True
    var_24 = 'stdio'
    var_25 = 'cstdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 indented from libc cimport stdio as cstdio'
    var_29 = 6
    var_30 = 'sys'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #19
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from collections import defaultdict\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from pathlib import Path as P\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'cimport numpy\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'from typing import (\n'
    var_27 = '    List,\n'
    var_28 = '    Dict,\n'
    var_29 = ')\n'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = True
    var_35 = module_1.Config()
    var_36 = 'import numpy as numpy\n'
    var_37 = [var_36]
    var_38 = module_0.imports(var_37, var_35)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = '    import os\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '/test/path.py'
    var_47 = [var_0]
    var_48 = len(var_44)
    assert var_48 == 1
    var_49 = 'def foo():\n'
    var_50 = '    import sys\n'
    var_51 = [var_0, var_49, var_50]
    var_52 = module_0.imports(var_51, top_only=var_34)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1



# Parsed testcases at query #20
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == ':1 import os'
    var_6 = 2
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 import numpy as np'
    var_11 = 3
    var_12 = True
    var_13 = 'sys'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 indented import sys'
    var_16 = 4
    var_17 = 'pandas'
    var_18 = 'test.py'
    var_19 = str(var_14)
    assert var_19 == 'test.py:4 import pandas'
    var_20 = 5
    var_21 = 'cython'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':5 cimport cython'
    var_25 = 6
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 from collections import defaultdict'
    var_30 = 7
    var_31 = 'dd'
    var_32 = module_0.Import()
    var_33 = str(var_32)
    assert var_33 == ':7 from collections import defaultdict as dd'
    var_34 = 8
    var_35 = True
    var_36 = 'os.path'
    var_37 = 'join'
    var_38 = 'script.py'
    var_39 = str(var_32)
    assert var_39 == 'script.py:8 indented from os.path import join'



# Parsed testcases at query #21
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'from os import path\n'
    var_7 = 'from sys import argv\n'
    var_8 = [var_6, var_7]
    var_9 = module_0.imports(var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'import numpy as np\n'
    var_13 = 'import pandas as pd\n'
    var_14 = [var_12, var_13]
    var_15 = module_0.imports(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'from os import path as p\n'
    var_19 = 'from sys import argv as a\n'
    var_20 = [var_18, var_19]
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'cimport numpy as np\n'
    var_25 = 'cimport pandas as pd\n'
    var_26 = [var_24, var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from os import (\n'
    var_31 = '    path,\n'
    var_32 = '    environ\n'
    var_33 = ')\n'
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = '    import os\n'
    var_39 = '    import sys\n'
    var_40 = [var_38, var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'import os  # comment\n'
    var_45 = 'import sys  # comment\n'
    var_46 = [var_44, var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = []
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 0
    var_54 = 'x = 1\n'
    var_55 = 'y = 2\n'
    var_56 = [var_54, var_55]
    var_57 = module_0.imports(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 0
    var_60 = [var_0, var_54, var_1]
    var_61 = module_0.imports(var_60)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = 'def foo():\n'
    var_65 = [var_0, var_64, var_39]
    var_66 = True
    var_67 = module_0.imports(var_65, top_only=var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1
    var_70 = [var_0, var_1]
    var_71 = '/path/to/file.py'
    var_72 = len(var_68)
    assert var_72 == 2
    var_73 = module_1.Config()
    var_74 = 'import os as os\n'
    var_75 = 'import sys as sys\n'
    var_76 = [var_74, var_75]
    var_77 = module_0.imports(var_76, var_73)
    var_78 = list(var_77)
    var_79 = len(var_78)
    assert var_79 == 2



# Parsed testcases at query #22
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'path'
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'from os import path'
    var_12 = 'p'
    var_13 = module_0.Import()
    var_14 = var_13.statement()
    assert var_14 == 'from os import path as p'
    var_15 = 'cython'
    var_16 = True
    var_17 = module_0.Import()
    var_18 = var_17.statement()
    assert var_18 == 'cimport cython'
    var_19 = 'cy'
    var_20 = True
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'cimport cython as cy'
    var_23 = 'view'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport view'
    var_27 = 'v'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport view as v'



# Parsed testcases at query #23
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'numpy'
    var_6 = 'np'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import numpy as np'
    var_9 = 'cython'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport cython'
    var_13 = True
    var_14 = 'cy'
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport cython as cy'
    var_17 = 'path'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path'
    var_20 = 'view'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport view'
    var_24 = 'p'
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = True
    var_28 = 'v'
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport view as v'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'module'
    var_3 = 'alias'
    var_4 = True
    var_5 = True
    var_6 = 'attribute'
    var_7 = True
    var_8 = True



# Parsed testcases at query #25
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import os'
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = var_8.statement()
    assert var_9 == 'import numpy as np'
    var_10 = 'path'
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'from os import path'
    var_13 = 'p'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'from os import path as p'
    var_16 = 'cython'
    var_17 = True
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'cimport cython'
    var_20 = 'cy'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'cimport cython as cy'
    var_24 = 'view'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from cython cimport view'
    var_28 = 'v'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport view as v'



