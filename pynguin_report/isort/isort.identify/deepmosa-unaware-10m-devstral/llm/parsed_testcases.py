####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_45 = 'from typing import (\n'
    var_46 = '    List,\n'
    var_47 = '    Dict,\n'
    var_48 = ')\n'
    var_49 = [var_45, var_46, var_47, var_48]
    var_50 = 'Dict'
    var_51 = module_0.imports(var_49)
    var_52 = list(var_51)
    var_53 = 'def foo():\n'
    var_54 = '    import os\n'
    var_55 = '    import sys\n'
    var_56 = [var_53, var_54, var_55]
    var_57 = True
    var_58 = 3
    var_59 = True
    var_60 = module_0.imports(var_56)
    var_61 = list(var_60)
    var_62 = 'x = 1\n'
    var_63 = 'y = 2\n'
    var_64 = [var_62, var_0, var_63, var_1]
    var_65 = 4
    var_66 = module_0.imports(var_64)
    var_67 = list(var_66)
    var_68 = 'import os  # Operating system\n'
    var_69 = 'import sys  # System\n'
    var_70 = [var_68, var_69]
    var_71 = module_0.imports(var_70)
    var_72 = list(var_71)
    var_73 = 'import os; import sys\n'
    var_74 = [var_73]
    var_75 = module_0.imports(var_74)
    var_76 = list(var_75)



# Parsed testcases at query #2
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
    var_23 = 'cfunc'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport cfunc'
    var_27 = 'cf'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #3
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
    var_10 = 'from sys import argv\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'import os, sys\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'from collections import (\n    OrderedDict,\n    defaultdict,\n)\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = True
    var_31 = module_1.Config()
    var_32 = 'import numpy as numpy\n'
    var_33 = [var_32]
    var_34 = module_0.imports(var_33, var_31)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = '    import os\n'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = '# This is a comment\nimport os  # inline comment\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 'import os\n\ndef foo():\n    pass\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48, top_only=var_30)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = '/test/file.py'
    var_53 = [var_0]
    var_54 = len(var_50)
    assert var_54 == 1
    var_55 = False
    var_56 = 'os'
    var_57 = None
    var_58 = 'path'
    var_59 = 'osp'
    var_60 = 'numpy'
    var_61 = 'np'
    var_62 = '/test.py'



# Parsed testcases at query #4
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
    var_15 = 'from pandas import DataFrame as DF\n'
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
    var_30 = 'from libc cimport printf\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from os import (\n    path,\n    environ\n)\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '    import sys\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = '/test/file.py'
    var_46 = [var_0]
    var_47 = len(var_43)
    assert var_47 == 1
    var_48 = 'def func():\n'
    var_49 = [var_0, var_48, var_40]
    var_50 = True
    var_51 = module_0.imports(var_49, top_only=var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = module_1.Config()
    var_55 = 'import os as os\n'
    var_56 = [var_55]
    var_57 = module_0.imports(var_56, var_54)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = 'import os  # comment\n'
    var_61 = [var_60]
    var_62 = module_0.imports(var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = 'import os; import sys\n'
    var_66 = [var_65]
    var_67 = module_0.imports(var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = 'from os import path, \\\n    environ\n'
    var_71 = [var_70]
    var_72 = module_0.imports(var_71)
    var_73 = list(var_72)
    var_74 = len(var_73)
    assert var_74 == 2
    var_75 = 'from os import (\n    path, \\\n    environ\n)\n'
    var_76 = [var_75]
    var_77 = module_0.imports(var_76)
    var_78 = list(var_77)
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = 'yield\n'
    var_81 = [var_80, var_0]
    var_82 = module_0.imports(var_81)
    var_83 = list(var_82)
    var_84 = len(var_83)
    assert var_84 == 1
    var_85 = 'raise ValueError\n'
    var_86 = [var_85, var_0]
    var_87 = module_0.imports(var_86)
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 1



# Parsed testcases at query #5
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
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'libc'
    var_22 = 'stdio'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from libc cimport stdio'
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'dd'
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from collections import defaultdict as dd'
    var_31 = 'math'
    var_32 = 'cm'
    var_33 = True
    var_34 = module_0.Import()
    var_35 = var_34.statement()
    assert var_35 == 'from libc cimport math as cm'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'libc'
    var_22 = 'stdio'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from libc cimport stdio'
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'dd'
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from collections import defaultdict as dd'
    var_31 = 'math'
    var_32 = 'cm'
    var_33 = True
    var_34 = module_0.Import()
    var_35 = var_34.statement()
    assert var_35 == 'from libc cimport math as cm'



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



# Parsed testcases at query #9
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = 1
    var_3 = False
    var_4 = 'os'
    var_5 = module_0.Import()
    var_6 = [var_5]
    var_7 = module_0.imports(var_1)
    var_8 = list(var_7)
    var_9 = 'import numpy as np'
    var_10 = [var_9]
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = [var_13]
    var_15 = module_0.imports(var_10)
    var_16 = list(var_15)
    var_17 = 'from sys import argv'
    var_18 = [var_17]
    var_19 = 'sys'
    var_20 = 'argv'
    var_21 = module_0.Import()
    var_22 = [var_21]
    var_23 = module_0.imports(var_18)
    var_24 = list(var_23)
    var_25 = 'from collections import OrderedDict as OD'
    var_26 = [var_25]
    var_27 = 'collections'
    var_28 = 'OrderedDict'
    var_29 = 'OD'
    var_30 = module_0.Import()
    var_31 = [var_30]
    var_32 = module_0.imports(var_26)
    var_33 = list(var_32)
    var_34 = 'cimport numpy'
    var_35 = [var_34]
    var_36 = True
    var_37 = module_0.Import()
    var_38 = [var_37]
    var_39 = module_0.imports(var_35)
    var_40 = list(var_39)
    var_41 = 'import os, sys'
    var_42 = [var_41]
    var_43 = module_0.Import()
    var_44 = module_0.Import()
    var_45 = [var_43, var_44]
    var_46 = module_0.imports(var_42)
    var_47 = list(var_46)
    var_48 = 'from typing import (\n    List,\n    Dict\n)'
    var_49 = [var_48]
    var_50 = 'typing'
    var_51 = 'List'
    var_52 = module_0.Import()
    var_53 = 'Dict'
    var_54 = module_0.Import()
    var_55 = [var_52, var_54]
    var_56 = module_0.imports(var_49)
    var_57 = list(var_56)
    var_58 = '    import os'
    var_59 = [var_58]
    var_60 = True
    var_61 = module_0.Import()
    var_62 = [var_61]
    var_63 = module_0.imports(var_59)
    var_64 = list(var_63)
    var_65 = 'import os  # comment'
    var_66 = [var_65]
    var_67 = module_0.Import()
    var_68 = [var_67]
    var_69 = module_0.imports(var_66)
    var_70 = list(var_69)
    var_71 = 'import os; import sys'
    var_72 = [var_71]
    var_73 = module_0.Import()
    var_74 = module_0.Import()
    var_75 = [var_73, var_74]
    var_76 = module_0.imports(var_72)
    var_77 = list(var_76)
    var_78 = ''
    var_79 = [var_78, var_0]
    var_80 = 2
    var_81 = module_0.Import()
    var_82 = [var_81]
    var_83 = module_0.imports(var_79)
    var_84 = list(var_83)
    var_85 = '# comment'
    var_86 = [var_85, var_0]
    var_87 = module_0.Import()
    var_88 = [var_87]
    var_89 = module_0.imports(var_86)
    var_90 = list(var_89)
    var_91 = 'from typing import \\\n    List'
    var_92 = [var_91]
    var_93 = module_0.Import()
    var_94 = [var_93]
    var_95 = module_0.imports(var_92)
    var_96 = list(var_95)
    var_97 = 'from typing import (\n    List,\n    Dict,\n)'
    var_98 = [var_97]
    var_99 = module_0.Import()
    var_100 = module_0.Import()
    var_101 = [var_99, var_100]
    var_102 = module_0.imports(var_98)
    var_103 = list(var_102)
    var_104 = True
    var_105 = module_1.Config()
    var_106 = 'import numpy as numpy'
    var_107 = [var_106]
    var_108 = module_0.Import()
    var_109 = [var_108]
    var_110 = module_0.imports(var_107, var_105)
    var_111 = list(var_110)
    var_112 = module_1.Config()
    var_113 = [var_106]
    var_114 = module_0.Import()
    var_115 = [var_114]
    var_116 = module_0.imports(var_113, var_112)
    var_117 = list(var_116)
    var_118 = '/path/to/file.py'
    var_119 = [var_0]
    var_120 = 'def func():'
    var_121 = '    import sys'
    var_122 = [var_0, var_120, var_121]
    var_123 = module_0.Import()
    var_124 = [var_123]
    var_125 = True
    var_126 = module_0.imports(var_122, top_only=var_125)
    var_127 = list(var_126)
    var_128 = 'import sys'
    var_129 = [var_0, var_128]
    var_130 = module_0.Import()
    var_131 = module_0.Import()
    var_132 = [var_130, var_131]
    var_133 = True
    var_134 = module_0.imports(var_129, top_only=var_133)
    var_135 = list(var_134)
    var_136 = 'yield'
    var_137 = [var_136, var_0]
    var_138 = module_0.Import()
    var_139 = [var_138]
    var_140 = module_0.imports(var_137)
    var_141 = list(var_140)
    var_142 = 'raise ValueError'
    var_143 = [var_142, var_0]
    var_144 = module_0.Import()
    var_145 = [var_144]
    var_146 = module_0.imports(var_143)
    var_147 = list(var_146)



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
    var_66 = 'from collections import defaultdict, \\\n'
    var_67 = [var_66, var_28]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = '    yield\n'
    var_72 = [var_49, var_71, var_41]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = 'raise Exception\n'
    var_77 = [var_76, var_0]
    var_78 = module_0.imports(var_77)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 1
    var_81 = []
    var_82 = module_0.imports(var_81)
    var_83 = list(var_82)
    var_84 = len(var_83)
    assert var_84 == 0



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
    var_10 = 'from sys import argv\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from os import path, environ\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'from libc cimport stdio\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'from os import (\n'
    var_31 = '    path,\n'
    var_32 = '    environ,\n'
    var_33 = ')\n'
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = 'import sys  # Some comment\n'
    var_39 = [var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = '    import os\n'
    var_44 = [var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = True
    var_49 = module_1.Config()
    var_50 = 'import os as os\n'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51, var_49)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'def foo():\n'
    var_56 = '    import sys\n'
    var_57 = [var_0, var_55, var_56]
    var_58 = module_0.imports(var_57, top_only=var_48)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = '/test/path'
    var_62 = [var_0]
    var_63 = len(var_59)
    assert var_63 == 1
    var_64 = 'from os import path, \\\n'
    var_65 = '    environ\n'
    var_66 = [var_64, var_65]
    var_67 = module_0.imports(var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 2



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
    var_18 = 'os.path'
    var_19 = 'join'
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'from os.path import join'
    var_22 = 'libc'
    var_23 = 'stdio'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from libc cimport stdio'
    var_27 = 'collections'
    var_28 = 'defaultdict'
    var_29 = 'dd'
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from collections import defaultdict as dd'
    var_32 = 'math'
    var_33 = 'cm'
    var_34 = True
    var_35 = module_0.Import()
    var_36 = var_35.statement()
    assert var_36 == 'from libc cimport math as cm'



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------


import isort.identify as module_0

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
    var_26 = 'import os, sys, json\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 3
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
    var_49 = 'def foo():\n'
    var_50 = '    pass\n'
    var_51 = [var_49, var_50, var_0]
    var_52 = module_0.imports(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = '    import sys\n'
    var_56 = [var_0, var_49, var_55]
    var_57 = True
    var_58 = module_0.imports(var_56, top_only=var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = [var_0]
    var_62 = '/path/to/file.py'
    var_63 = len(var_59)
    assert var_63 == 1



# Parsed testcases at query #16
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
    var_17 = 'os.path'
    var_18 = 'join'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os.path import join'
    var_21 = 'path_join'
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from os.path import join as path_join'
    var_24 = 'cdef'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from cython cimport cdef'
    var_28 = 'cd'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport cdef as cd'



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
    var_17 = 'typing'
    var_18 = 'List'
    var_19 = 'MyList'
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'from typing import List as MyList'
    var_22 = 5
    var_23 = 'cython'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'cimport cython'
    var_27 = 6
    var_28 = 'cy'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'cimport cython as cy'
    var_32 = 7
    var_33 = 'cdef'
    var_34 = True
    var_35 = module_0.Import()
    var_36 = var_35.statement()
    assert var_36 == 'from cython cimport cdef'
    var_37 = 8
    var_38 = 'cd'
    var_39 = True
    var_40 = module_0.Import()
    var_41 = var_40.statement()
    assert var_41 == 'from cython cimport cdef as cd'



# Parsed testcases at query #18
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
    var_6 = 2
    var_7 = 'sys'
    var_8 = module_0.imports(var_2)
    var_9 = list(var_8)
    var_10 = 'import numpy as np\n'
    var_11 = [var_10]
    var_12 = 'numpy'
    var_13 = 'np'
    var_14 = module_0.imports(var_11)
    var_15 = list(var_14)
    var_16 = 'from collections import defaultdict\n'
    var_17 = [var_16]
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = module_0.imports(var_17)
    var_21 = list(var_20)
    var_22 = 'from pathlib import Path as P\n'
    var_23 = [var_22]
    var_24 = 'pathlib'
    var_25 = 'Path'
    var_26 = 'P'
    var_27 = module_0.imports(var_23)
    var_28 = list(var_27)
    var_29 = 'cimport numpy\n'
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.imports(var_30)
    var_33 = list(var_32)
    var_34 = 'import os, sys\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = '    import os\n'
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.imports(var_39)
    var_42 = list(var_41)
    var_43 = 'from collections import (\n'
    var_44 = '    defaultdict,\n'
    var_45 = '    Counter\n'
    var_46 = ')\n'
    var_47 = [var_43, var_44, var_45, var_46]
    var_48 = 'Counter'
    var_49 = module_0.imports(var_47)
    var_50 = list(var_49)
    var_51 = 'import os  # Operating system\n'
    var_52 = [var_51]
    var_53 = module_0.imports(var_52)
    var_54 = list(var_53)
    var_55 = 'x = 1\n'
    var_56 = 'y = 2\n'
    var_57 = [var_55, var_0, var_56]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = '/test.py'
    var_61 = [var_0]
    var_62 = 'def func():\n'
    var_63 = '    pass\n'
    var_64 = [var_0, var_62, var_63]
    var_65 = True
    var_66 = module_0.imports(var_64, top_only=var_65)
    var_67 = list(var_66)
    var_68 = True
    var_69 = module_1.Config()
    var_70 = 'import numpy as numpy\n'
    var_71 = [var_70]
    var_72 = module_0.imports(var_71, var_69)
    var_73 = list(var_72)
    var_74 = True
    var_75 = module_1.Config()
    var_76 = 'from os import path as path\n'
    var_77 = [var_76]
    var_78 = 'path'
    var_79 = module_0.imports(var_77, var_75)
    var_80 = list(var_79)



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
    var_6 = 1
    var_7 = False
    var_8 = 'os'
    var_9 = None
    var_10 = 2
    var_11 = 'sys'
    var_12 = 'import numpy as np\n'
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'numpy'
    var_18 = 'np'
    var_19 = 'from collections import defaultdict\n'
    var_20 = [var_19]
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'collections'
    var_25 = 'defaultdict'
    var_26 = 'from pathlib import Path as P\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'pathlib'
    var_32 = 'Path'
    var_33 = 'P'
    var_34 = 'cimport numpy\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = 'from cython cimport int\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'cython'
    var_46 = 'int'
    var_47 = True
    var_48 = 'from collections import (\n'
    var_49 = '    defaultdict,\n'
    var_50 = '    Counter\n'
    var_51 = ')\n'
    var_52 = [var_48, var_49, var_50, var_51]
    var_53 = module_0.imports(var_52)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = 'Counter'
    var_57 = '    import os\n'
    var_58 = [var_57]
    var_59 = module_0.imports(var_58)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 1
    var_62 = True
    var_63 = '/tmp/test.py'
    var_64 = [var_0]
    var_65 = len(var_60)
    assert var_65 == 1
    var_66 = 'def foo():\n'
    var_67 = '    import sys\n'
    var_68 = [var_0, var_66, var_67]
    var_69 = True
    var_70 = module_0.imports(var_68, top_only=var_69)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 1
    var_73 = True
    var_74 = module_1.Config()
    var_75 = 'import os as os\n'
    var_76 = [var_75]
    var_77 = module_0.imports(var_76, var_74)
    var_78 = list(var_77)
    var_79 = len(var_78)
    assert var_79 == 1
    var_80 = 'import os  # comment\n'
    var_81 = [var_80]
    var_82 = module_0.imports(var_81)
    var_83 = list(var_82)
    var_84 = len(var_83)
    assert var_84 == 1
    var_85 = 'import os; import sys\n'
    var_86 = [var_85]
    var_87 = module_0.imports(var_86)
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = 'import os \\\n'
    var_91 = '    , sys\n'
    var_92 = [var_90, var_91]
    var_93 = module_0.imports(var_92)
    var_94 = list(var_93)
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = 'yield\n'
    var_97 = [var_96, var_0]
    var_98 = module_0.imports(var_97)
    var_99 = list(var_98)
    var_100 = len(var_99)
    assert var_100 == 1
    var_101 = 'raise\n'
    var_102 = [var_101, var_0]
    var_103 = module_0.imports(var_102)
    var_104 = list(var_103)
    var_105 = len(var_104)
    assert var_105 == 1
    var_106 = '\n'
    var_107 = [var_106, var_0, var_106]
    var_108 = module_0.imports(var_107)
    var_109 = list(var_108)
    var_110 = len(var_109)
    assert var_110 == 1
    var_111 = '# isort: on'
    var_112 = '# isort: off'
    var_113 = [var_111, var_112]
    var_114 = module_1.Config()
    var_115 = '# isort: off\n'
    var_116 = '# isort: on\n'
    var_117 = [var_115, var_0, var_116, var_1]
    var_118 = module_0.imports(var_117, var_114)
    var_119 = list(var_118)
    var_120 = len(var_119)
    assert var_120 == 2
    var_121 = 4



# Parsed testcases at query #20
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
    var_20 = 'cimport numpy'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import sys, os'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from typing import (\n    List,\n    Dict,\n)'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import os'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = 'import os  # some comment'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'import os; import sys'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = True
    var_51 = module_1.Config()
    var_52 = 'import numpy as numpy'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53, var_51)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/some/path/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = 'def foo():'
    var_61 = 'import sys'
    var_62 = [var_0, var_60, var_61]
    var_63 = module_0.imports(var_62, top_only=var_50)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'print("import os")'
    var_67 = [var_66, var_61]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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
    var_9 = True
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented cimport numpy'
    var_12 = 3
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = 'dd'
    var_16 = module_0.Import()
    var_17 = str(var_16)
    assert var_17 == ':3 from collections import defaultdict as dd'
    var_18 = 4
    var_19 = True
    var_20 = 'libc'
    var_21 = 'stdio'
    var_22 = 'cstdio'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':4 indented from libc cimport stdio as cstdio'
    var_26 = 5
    var_27 = 'sys'
    var_28 = '/path/to/file.py'
    var_29 = str(var_24)
    assert var_29 == '/path/to/file.py:5 import sys'



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



# Parsed testcases at query #24
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'operating_system'
    var_6 = module_0.Import()
    var_7 = var_6.statement()
    assert var_7 == 'import os as operating_system'
    var_8 = 'path'
    var_9 = module_0.Import()
    var_10 = var_9.statement()
    assert var_10 == 'from os import path'
    var_11 = 'p'
    var_12 = module_0.Import()
    var_13 = var_12.statement()
    assert var_13 == 'from os import path as p'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport os'
    var_17 = True
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'cimport os as operating_system'
    var_20 = True
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from os cimport path'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from os cimport path as p'



# Parsed testcases at query #25
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
    var_29 = True
    var_30 = 'cy'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = str(var_32)
    assert var_33 == ':6 indented cimport cython as cy'
    var_34 = 7
    var_35 = 'view'
    var_36 = True
    var_37 = module_0.Import()
    var_38 = str(var_37)
    assert var_38 == ':7 from cython cimport view'
    var_39 = 8
    var_40 = True
    var_41 = 'cv'
    var_42 = True
    var_43 = module_0.Import()
    var_44 = str(var_43)
    assert var_44 == ':8 indented from cython cimport view as cv'
    var_45 = 9
    var_46 = '/path/to/file.py'
    var_47 = str(var_43)
    assert var_47 == '/path/to/file.py:9 import sys'



# Parsed testcases at query #26
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
    var_17 = True
    var_18 = 'sys'
    var_19 = 'path'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 indented from sys import path'
    var_22 = 5
    var_23 = 'libc'
    var_24 = 'stdio'
    var_25 = 'cstdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 from libc cimport stdio as cstdio'
    var_29 = 6
    var_30 = True
    var_31 = 'pytest'
    var_32 = '/project/test.py'
    var_33 = str(var_27)
    assert var_33 == '/project/test.py:6 indented import pytest'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
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
    assert var_11 == ':5 import sys'
    var_12 = 15
    var_13 = 'numpy'
    var_14 = 'test.py'
    var_15 = str(var_10)
    assert var_15 == 'test.py:15 indented cimport numpy'
    var_16 = 20
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'example.py'
    var_20 = str(var_10)
    assert var_20 == 'example.py:20 from collections import defaultdict'



# Parsed testcases at query #29
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
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from cython cimport cython'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'from cython cimport cython as cy'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'numpy'
    var_4 = 'np'
    var_5 = 'cython'
    var_6 = True
    var_7 = 'cy'
    var_8 = True
    var_9 = 'path'
    var_10 = 'p'
    var_11 = 'view'
    var_12 = True
    var_13 = 'v'
    var_14 = True



# Parsed testcases at query #31
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
    var_30 = 'from libcpp cimport bool\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from os import (\n'
    var_36 = '    path,\n'
    var_37 = '    environ\n'
    var_38 = ')\n'
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = '    import sys\n'
    var_44 = [var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = True
    var_49 = module_1.Config()
    var_50 = 'import numpy as numpy\n'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51, var_49)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'import os  # comment\n'
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
    var_65 = 'def foo():\n'
    var_66 = [var_0, var_65, var_43]
    var_67 = module_0.imports(var_66, top_only=var_48)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1



# Parsed testcases at query #32
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
    assert var_11 == ':5 import sys'
    var_12 = 15
    var_13 = 'libc'
    var_14 = 'stdio'
    var_15 = '/test/cython.py'
    var_16 = str(var_10)
    assert var_16 == '/test/cython.py:15 indented from libc cimport stdio'
    var_17 = 20
    var_18 = 'numpy'
    var_19 = 'np'
    var_20 = '/test/numpy.py'
    var_21 = str(var_10)
    assert var_21 == '/test/numpy.py:20 import numpy as np'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'numpy'
    var_4 = 'np'
    var_5 = 'cython'
    var_6 = True
    var_7 = 'cy'
    var_8 = True
    var_9 = 'path'
    var_10 = 'libc'
    var_11 = 'stdio'
    var_12 = True
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = 'dd'
    var_16 = 'math'
    var_17 = 'lm'
    var_18 = True



# Parsed testcases at query #34
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
    var_30 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import sys\n'
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
    var_45 = True
    var_46 = module_1.Config()
    var_47 = 'import numpy as numpy\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48, var_46)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'def foo():\n'
    var_53 = [var_0, var_52, var_35]
    var_54 = module_0.imports(var_53, top_only=var_45)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/test/path'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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



# Parsed testcases at query #37
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
    var_10 = 'from os import path\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from os import path as p\n'
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
    var_30 = 'from os import (\n    path,\n    environ\n)\n'
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
    var_40 = 'import os  # comment\n'
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
    var_50 = True
    var_51 = module_1.Config()
    var_52 = 'import numpy as numpy\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53, var_51)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/test/path'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = 'def func():\n'
    var_61 = '    pass\n'
    var_62 = [var_0, var_60, var_61]
    var_63 = module_0.imports(var_62, top_only=var_50)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = '# comment\n'
    var_67 = [var_66, var_0]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = 'yield\n'
    var_72 = [var_71, var_0]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = 'import os \\\n'
    var_77 = '    , sys\n'
    var_78 = [var_76, var_77]
    var_79 = module_0.imports(var_78)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 2



# Parsed testcases at query #38
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
    var_8 = 'sys'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import sys'
    var_11 = 3
    var_12 = 'numpy'
    var_13 = 'array'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from numpy import array'
    var_16 = 4
    var_17 = 'pandas'
    var_18 = 'pd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 import pandas as pd'
    var_21 = 5
    var_22 = True
    var_23 = 'tensorflow'
    var_24 = 'keras'
    var_25 = 'tfk'
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':5 indented from tensorflow import keras as tfk'
    var_28 = 6
    var_29 = 'cython'
    var_30 = True
    var_31 = module_0.Import()
    var_32 = str(var_31)
    assert var_32 == ':6 cimport cython'
    var_33 = 7
    var_34 = 'django'
    var_35 = '/project/main.py'
    var_36 = str(var_31)
    assert var_36 == '/project/main.py:7 import django'
    var_37 = 8
    var_38 = True
    var_39 = 'scipy'
    var_40 = 'stats'
    var_41 = 'spst'
    var_42 = True
    var_43 = '/project/utils.py'
    var_44 = str(var_31)
    assert var_44 == '/project/utils.py:8 indented from scipy cimport stats as spst'



# Parsed testcases at query #39
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
    var_25 = 'c_stdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 indented from libc cimport stdio as c_stdio'
    var_29 = 6
    var_30 = 'sys'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #40
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'operating_system'
    var_6 = module_0.Import()
    var_7 = var_6.statement()
    assert var_7 == 'import os as operating_system'
    var_8 = 'cython'
    var_9 = True
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'cimport cython'
    var_12 = 'cy'
    var_13 = True
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'cimport cython as cy'
    var_16 = 'os.path'
    var_17 = 'join'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os.path import join'
    var_20 = 'path_join'
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from os.path import join as path_join'
    var_23 = 'cdivision'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport cdivision'
    var_27 = 'cd'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cdivision as cd'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = 2
    var_5 = True
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = 3
    var_9 = 'sys'
    var_10 = 'path'
    var_11 = 4
    var_12 = 'cython'
    var_13 = True
    var_14 = 5
    var_15 = 'module'
    var_16 = '/path/to/file.py'
    var_17 = 6
    var_18 = True
    var_19 = 'attr'
    var_20 = 'alias'
    var_21 = True



# Parsed testcases at query #42
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = False
    var_6 = '/some/path/file.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = 'exit'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':5 from sys cimport exit'
    var_13 = 'json'
    var_14 = 'js'
    var_15 = 'test.py'
    var_16 = str(var_11)
    assert var_16 == 'test.py:1 import json as js'
    var_17 = 20
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':20 indented from collections import defaultdict'
    var_22 = 3
    var_23 = 'cython'
    var_24 = 'module.pyx'
    var_25 = str(var_20)
    assert var_25 == 'module.pyx:3 cimport cython'



# Parsed testcases at query #43
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'module'
    var_3 = 'attribute'
    var_4 = 'alias'
    var_5 = True
    var_6 = 'path/to/file.py'
    var_7 = 2
    var_8 = True
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':2 indented import from module attribute'
    var_12 = 3
    var_13 = 'another/path.py'
    var_14 = str(var_10)
    assert var_14 == 'another/path.py:3 import module'



# Parsed testcases at query #44
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
    var_6 = 'operating_system'
    var_7 = module_0.Import()
    var_8 = var_7.statement()
    assert var_8 == 'import os as operating_system'
    var_9 = 'numpy'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport numpy'
    var_13 = 'np'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport numpy as np'
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from collections import defaultdict'
    var_21 = 'dd'
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from collections import defaultdict as dd'
    var_24 = 'libc'
    var_25 = 'stdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'from libc cimport stdio'
    var_29 = 'cstdio'
    var_30 = True
    var_31 = module_0.Import()
    var_32 = var_31.statement()
    assert var_32 == 'from libc cimport stdio as cstdio'



# Parsed testcases at query #45
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



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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
    var_20 = 'collections'
    var_21 = 'defaultdict'
    var_22 = 'dd'
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from collections import defaultdict as dd'
    var_25 = 'cdef_class'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'from cython cimport cdef_class'
    var_29 = 'cc'
    var_30 = True
    var_31 = module_0.Import()
    var_32 = var_31.statement()
    assert var_32 == 'from cython cimport cdef_class as cc'



# Parsed testcases at query #48
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
    var_7 = 'from os import path\n'
    var_8 = 'from sys import argv\n'
    var_9 = [var_7, var_8]
    var_10 = iter(var_9)
    var_11 = module_0.imports(var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'import numpy as np\n'
    var_15 = 'import pandas as pd\n'
    var_16 = [var_14, var_15]
    var_17 = iter(var_16)
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = 'from os import path as p\n'
    var_22 = 'from sys import argv as a\n'
    var_23 = [var_21, var_22]
    var_24 = iter(var_23)
    var_25 = module_0.imports(var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = 'cimport numpy\n'
    var_29 = 'from numpy cimport array\n'
    var_30 = [var_28, var_29]
    var_31 = iter(var_30)
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'from os import (\n'
    var_36 = '    path,\n'
    var_37 = '    environ\n'
    var_38 = ')\n'
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = iter(var_39)
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = '    import os\n'
    var_45 = [var_44, var_1]
    var_46 = iter(var_45)
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = '/test/path'
    var_51 = [var_0]
    var_52 = iter(var_51)
    var_53 = len(var_48)
    assert var_53 == 1
    var_54 = 'def function():\n'
    var_55 = '    import sys\n'
    var_56 = [var_0, var_54, var_55]
    var_57 = iter(var_56)
    var_58 = True
    var_59 = module_0.imports(var_57, top_only=var_58)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 1
    var_62 = module_1.Config()
    var_63 = 'import os as os\n'
    var_64 = 'from sys import argv as argv\n'
    var_65 = [var_63, var_64]
    var_66 = iter(var_65)
    var_67 = module_0.imports(var_66, var_62)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = '# Comment\n'
    var_71 = 'import os  # Comment\n'
    var_72 = [var_70, var_71, var_1]
    var_73 = iter(var_72)
    var_74 = module_0.imports(var_73)
    var_75 = list(var_74)
    var_76 = len(var_75)
    assert var_76 == 2
    var_77 = 'import os; import sys\n'
    var_78 = [var_77]
    var_79 = iter(var_78)
    var_80 = module_0.imports(var_79)
    var_81 = list(var_80)
    var_82 = len(var_81)
    assert var_82 == 2
    var_83 = 'import os \\\n'
    var_84 = '    , sys\n'
    var_85 = [var_83, var_84]
    var_86 = iter(var_85)
    var_87 = module_0.imports(var_86)
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = '    yield\n'
    var_91 = [var_54, var_90, var_44]
    var_92 = iter(var_91)
    var_93 = module_0.imports(var_92)
    var_94 = list(var_93)
    var_95 = len(var_94)
    assert var_95 == 1
    var_96 = '    raise\n'
    var_97 = [var_54, var_96, var_44]
    var_98 = iter(var_97)
    var_99 = module_0.imports(var_98)
    var_100 = list(var_99)
    var_101 = len(var_100)
    assert var_101 == 1



# Parsed testcases at query #49
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'operating_system'
    var_6 = module_0.Import()
    var_7 = var_6.statement()
    assert var_7 == 'import os as operating_system'
    var_8 = 'numpy'
    var_9 = True
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'cimport numpy'
    var_12 = True
    var_13 = 'np'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'cimport numpy as np'
    var_16 = 'path'
    var_17 = module_0.Import()
    var_18 = var_17.statement()
    assert var_18 == 'from os import path'
    var_19 = 'libc'
    var_20 = 'stdio'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from libc cimport stdio'
    var_24 = 'p'
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = True
    var_28 = 's'
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from libc cimport stdio as s'



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'sys'
    var_3 = 'numpy'
    var_4 = 'np'
    var_5 = 'cython'
    var_6 = True
    var_7 = 'cy'
    var_8 = True
    var_9 = 'os'
    var_10 = 'path'
    var_11 = 'collections'
    var_12 = 'defaultdict'
    var_13 = 'dd'
    var_14 = 'libc'
    var_15 = 'stdio'
    var_16 = True
    var_17 = 'math'
    var_18 = 'lm'
    var_19 = True



# Parsed testcases at query #52
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
    var_15 = 'from os import path, environ'
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
    var_25 = 'from os import (\n    path,\n    environ\n)'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'import os  # This is a comment'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = '    import os'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = True
    var_41 = module_1.Config()
    var_42 = 'import os as os'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43, var_41)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 'def foo():'
    var_48 = '    import sys'
    var_49 = [var_0, var_47, var_48]
    var_50 = module_0.imports(var_49, top_only=var_40)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 1
    var_53 = '/test/file.py'
    var_54 = [var_0]
    var_55 = len(var_51)
    assert var_55 == 1



# Parsed testcases at query #53
#--------------------------


import isort.identify as module_0

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
    var_25 = 'cimport numpy\n'
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
    var_40 = 'import os  # comment\n'
    var_41 = [var_40]
    var_42 = iter(var_41)
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = []
    var_47 = iter(var_46)
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 0
    var_51 = 'x = 1\n'
    var_52 = 'def foo():\n'
    var_53 = '    pass\n'
    var_54 = [var_51, var_52, var_53]
    var_55 = iter(var_54)
    var_56 = module_0.imports(var_55)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 0
    var_59 = '    import sys\n'
    var_60 = [var_0, var_52, var_59]
    var_61 = iter(var_60)
    var_62 = True
    var_63 = module_0.imports(var_61, top_only=var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1



# Parsed testcases at query #54
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
    var_9 = 'libc'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport libc'
    var_13 = 'c'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport libc as c'
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
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



# Parsed testcases at query #55
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
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict'
    var_21 = 5
    var_22 = 'cython'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 cimport cython'
    var_26 = 6
    var_27 = 'libc'
    var_28 = 'stdio'
    var_29 = 'c_stdio'
    var_30 = True
    var_31 = module_0.Import()
    var_32 = str(var_31)
    assert var_32 == ':6 from libc cimport stdio as c_stdio'
    var_33 = 7
    var_34 = 'pathlib'
    var_35 = '/path/to/file.py'
    var_36 = str(var_31)
    assert var_36 == '/path/to/file.py:7 import pathlib'
    var_37 = 8
    var_38 = True
    var_39 = 'typing'
    var_40 = 'List'
    var_41 = '/another/path.py'
    var_42 = str(var_31)
    assert var_42 == '/another/path.py:8 indented from typing import List'



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'sys'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import sys'
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
    var_18 = 'os'
    var_19 = 'path'
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'from os import path'
    var_22 = 'libc'
    var_23 = 'stdio'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from libc cimport stdio'
    var_27 = 'collections'
    var_28 = 'defaultdict'
    var_29 = 'dd'
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from collections import defaultdict as dd'



# Parsed testcases at query #58
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
    var_20 = 'cmath'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport cmath'
    var_24 = 'p'
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = 'cm'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cmath as cm'



# Parsed testcases at query #59
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
    var_34 = 'import os  # Operating system interfaces\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = '# This is a comment\n'
    var_40 = '"""Docstring"""\n'
    var_41 = [var_39, var_0, var_40, var_1]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = 'if True:\n'
    var_46 = '    import os\n'
    var_47 = [var_45, var_46]
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = 'def function():\n'
    var_52 = '    pass\n'
    var_53 = [var_0, var_51, var_52, var_1]
    var_54 = True
    var_55 = module_0.imports(var_53, top_only=var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = module_1.Config()
    var_59 = 'import numpy as numpy\n'
    var_60 = [var_59]
    var_61 = module_0.imports(var_60, var_58)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1



# Parsed testcases at query #60
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



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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
    var_28 = '    Counter\n'
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
    var_46 = '# This is a comment\n'
    var_47 = 'import os  # inline comment\n'
    var_48 = [var_46, var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'import os; import sys\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = '/test/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = 'def function():\n'
    var_61 = '    import sys\n'
    var_62 = [var_0, var_60, var_61]
    var_63 = module_0.imports(var_62, top_only=var_34)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1



# Parsed testcases at query #63
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



# Parsed testcases at query #64
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
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from collections import defaultdict'
    var_16 = 4
    var_17 = True
    var_18 = 'typing'
    var_19 = 'List'
    var_20 = 'T'
    var_21 = module_0.Import()
    var_22 = str(var_21)
    assert var_22 == ':4 indented from typing import List as T'
    var_23 = 5
    var_24 = 'cython'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':5 cimport cython'
    var_28 = 6
    var_29 = 'sys'
    var_30 = '/path/to/file.py'
    var_31 = str(var_26)
    assert var_31 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #65
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



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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
    var_20 = 'from typing import List, Dict\n'
    var_21 = [var_19, var_20]
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = 'typing'
    var_25 = 'List'
    var_26 = 'Dict'
    var_27 = module_0.imports(var_21)
    var_28 = list(var_27)
    var_29 = 'from pathlib import Path as P\n'
    var_30 = 'from typing import List as L\n'
    var_31 = [var_29, var_30]
    var_32 = 'pathlib'
    var_33 = 'Path'
    var_34 = 'P'
    var_35 = 'L'
    var_36 = module_0.imports(var_31)
    var_37 = list(var_36)
    var_38 = 'cimport numpy as np\n'
    var_39 = 'from libc.math cimport sin\n'
    var_40 = [var_38, var_39]
    var_41 = True
    var_42 = 'libc.math'
    var_43 = 'sin'
    var_44 = True
    var_45 = module_0.imports(var_40)
    var_46 = list(var_45)
    var_47 = 'def foo():\n'
    var_48 = '    import os\n'
    var_49 = '    from sys import path\n'
    var_50 = [var_47, var_48, var_49]
    var_51 = True
    var_52 = 3
    var_53 = True
    var_54 = 'path'
    var_55 = module_0.imports(var_50)
    var_56 = list(var_55)
    var_57 = 'from typing import (\n'
    var_58 = '    List,\n'
    var_59 = '    Dict,\n'
    var_60 = ')\n'
    var_61 = [var_57, var_58, var_59, var_60]
    var_62 = module_0.imports(var_61)
    var_63 = list(var_62)
    var_64 = 'import os  # Operating system\n'
    var_65 = 'import sys  # System\n'
    var_66 = [var_64, var_65]
    var_67 = module_0.imports(var_66)
    var_68 = list(var_67)
    var_69 = 'x = 1\n'
    var_70 = 'y = 2\n'
    var_71 = 'from sys import path\n'
    var_72 = [var_69, var_0, var_70, var_71]
    var_73 = 4
    var_74 = module_0.imports(var_72)
    var_75 = list(var_74)
    var_76 = []
    var_77 = []
    var_78 = module_0.imports(var_76)
    var_79 = list(var_78)



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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



# Parsed testcases at query #70
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
    var_30 = 'from libcpp cimport bool'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from collections import (\n    defaultdict,\n    OrderedDict\n)'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '    import os'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = '/test/path.py'
    var_46 = [var_0]
    var_47 = len(var_43)
    assert var_47 == 1
    var_48 = True
    var_49 = module_1.Config()
    var_50 = 'import os as os'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51, var_49)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = 'def func():'
    var_56 = '    import sys'
    var_57 = [var_0, var_55, var_56]
    var_58 = module_0.imports(var_57, top_only=var_48)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = '"""docstring"""'
    var_62 = [var_61, var_0]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'import os  # some comment'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #71
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



# Parsed testcases at query #72
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
    var_47 = [var_1]
    var_48 = len(var_44)
    assert var_48 == 1
    var_49 = 'def func():\n'
    var_50 = '    pass\n'
    var_51 = [var_0, var_49, var_50]
    var_52 = module_0.imports(var_51, top_only=var_34)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1
    var_55 = '# comment\n'
    var_56 = [var_55, var_0]
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
    var_65 = 'from typing import \\\n'
    var_66 = '    List\n'
    var_67 = [var_65, var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #73
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
    var_23 = 'Path'
    var_24 = '/test.py'



# Parsed testcases at query #74
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
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = 'dd'
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from collections import defaultdict as dd'
    var_26 = 'cdef'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport cdef'
    var_30 = 'cd'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = var_32.statement()
    assert var_33 == 'from cython cimport cdef as cd'



# Parsed testcases at query #75
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
    var_17 = 'from os import path\n'
    var_18 = [var_17]
    var_19 = 'path'
    var_20 = module_0.imports(var_18)
    var_21 = list(var_20)
    var_22 = 'from os import path as p\n'
    var_23 = [var_22]
    var_24 = 'p'
    var_25 = module_0.imports(var_23)
    var_26 = list(var_25)
    var_27 = 'cimport numpy\n'
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.imports(var_28)
    var_31 = list(var_30)
    var_32 = 'from os import (\n'
    var_33 = '    path,\n'
    var_34 = '    environ\n'
    var_35 = ')\n'
    var_36 = [var_32, var_33, var_34, var_35]
    var_37 = 3
    var_38 = 'environ'
    var_39 = module_0.imports(var_36)
    var_40 = list(var_39)
    var_41 = '    import os\n'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.imports(var_42)
    var_45 = list(var_44)
    var_46 = True
    var_47 = module_1.Config()
    var_48 = 'import numpy as numpy\n'
    var_49 = [var_48]
    var_50 = module_0.imports(var_49, var_47)
    var_51 = list(var_50)
    var_52 = '# This is a comment\n'
    var_53 = 'import os  # inline comment\n'
    var_54 = [var_52, var_53]
    var_55 = module_0.imports(var_54)
    var_56 = list(var_55)
    var_57 = '/test/file.py'
    var_58 = [var_0]
    var_59 = 'def foo():\n'
    var_60 = '    import sys\n'
    var_61 = [var_0, var_59, var_60]
    var_62 = True
    var_63 = module_0.imports(var_61, top_only=var_62)
    var_64 = list(var_63)



# Parsed testcases at query #76
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'numpy'
    var_3 = 'array'
    var_4 = 'np_array'
    var_5 = True
    var_6 = 'test.py'
    var_7 = 2
    var_8 = True
    var_9 = 'os'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ' :2 indented import os'
    var_13 = 3
    var_14 = 'sys'
    var_15 = 'path'
    var_16 = 'example.py'
    var_17 = str(var_11)
    assert var_17 == 'example.py:3 import from sys path'
    var_18 = 4
    var_19 = True
    var_20 = 'pandas'
    var_21 = 'pd'
    var_22 = 'data.py'
    var_23 = str(var_11)
    assert var_23 == 'data.py:4 indented import pandas as pd'



# Parsed testcases at query #77
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
    var_25 = 'collections'
    var_26 = 'defaultdict'
    var_27 = 'dd'
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from collections import defaultdict as dd'
    var_30 = 'math'
    var_31 = 'lm'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = var_33.statement()
    assert var_34 == 'from libc cimport math as lm'



# Parsed testcases at query #78
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
    var_21 = 'T'
    var_22 = module_0.Import()
    var_23 = str(var_22)
    assert var_23 == ':4 indented from typing import List as T'
    var_24 = 5
    var_25 = 'libc'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 cimport libc'
    var_29 = 6
    var_30 = 'sys'
    var_31 = '/path/to/file.py'
    var_32 = str(var_27)
    assert var_32 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #79
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
    var_18 = 'from collections import OrderedDict as OD\n'
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
    var_36 = '    import os\n'
    var_37 = [var_36]
    var_38 = iter(var_37)
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = 'from os import (\n'
    var_43 = '    path,\n'
    var_44 = '    environ\n'
    var_45 = ')\n'
    var_46 = [var_42, var_43, var_44, var_45]
    var_47 = iter(var_46)
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 2
    var_51 = 'import os  # Comment\n'
    var_52 = [var_51]
    var_53 = iter(var_52)
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = 'x = 1\n'
    var_58 = 'y = 2\n'
    var_59 = [var_57, var_0, var_58]
    var_60 = iter(var_59)
    var_61 = module_0.imports(var_60)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'def func():\n'
    var_65 = '    import sys\n'
    var_66 = [var_0, var_64, var_65]
    var_67 = iter(var_66)
    var_68 = True
    var_69 = module_0.imports(var_67, top_only=var_68)
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 1
    var_72 = module_1.Config()
    var_73 = 'import numpy as numpy\n'
    var_74 = [var_73]
    var_75 = iter(var_74)
    var_76 = module_0.imports(var_75, var_72)
    var_77 = list(var_76)
    var_78 = len(var_77)
    assert var_78 == 1



# Parsed testcases at query #80
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



# Parsed testcases at query #81
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



# Parsed testcases at query #82
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



# Parsed testcases at query #83
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
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from collections import defaultdict'
    var_16 = 4
    var_17 = True
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 indented from collections import defaultdict as dd'
    var_21 = 5
    var_22 = 'cython'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 cimport cython'
    var_26 = 6
    var_27 = 'sys'
    var_28 = '/path/to/file.py'
    var_29 = str(var_24)
    assert var_29 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #84
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
    var_12 = 'from sys import argv\n'
    var_13 = [var_12]
    var_14 = iter(var_13)
    var_15 = module_0.imports(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 'from collections import OrderedDict as OD\n'
    var_19 = [var_18]
    var_20 = iter(var_19)
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'import os, sys\n'
    var_25 = [var_24]
    var_26 = iter(var_25)
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'cimport numpy\n'
    var_31 = [var_30]
    var_32 = iter(var_31)
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = 'from libc cimport printf\n'
    var_37 = [var_36]
    var_38 = iter(var_37)
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = 'from os import (\n    path,\n    sys\n)\n'
    var_43 = [var_42]
    var_44 = iter(var_43)
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = '    import os\n'
    var_49 = [var_48]
    var_50 = iter(var_49)
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = 'import os  # comment\n'
    var_55 = [var_54]
    var_56 = iter(var_55)
    var_57 = module_0.imports(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = 'import os; import sys\n'
    var_61 = [var_60]
    var_62 = iter(var_61)
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = True
    var_67 = module_1.Config()
    var_68 = 'import numpy as numpy\n'
    var_69 = [var_68]
    var_70 = iter(var_69)
    var_71 = module_0.imports(var_70, var_67)
    var_72 = list(var_71)
    var_73 = len(var_72)
    assert var_73 == 1
    var_74 = 'def foo():\n'
    var_75 = '    import sys\n'
    var_76 = [var_0, var_74, var_75]
    var_77 = iter(var_76)
    var_78 = module_0.imports(var_77, top_only=var_66)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 1
    var_81 = '/test.py'
    var_82 = [var_0]
    var_83 = iter(var_82)
    var_84 = len(var_79)
    assert var_84 == 1
    var_85 = 'import sys\n'
    var_86 = [var_0, var_85]
    var_87 = iter(var_86)
    var_88 = module_0.imports(var_87)
    var_89 = list(var_88)



# Parsed testcases at query #85
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
    var_11 = 'sys'
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 from sys import path'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict as dd'
    var_21 = 5
    var_22 = 'cython'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 cimport cython'
    var_26 = 6
    var_27 = True
    var_28 = 'typing'
    var_29 = 'List'
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':6 indented from typing import List'
    var_32 = 7
    var_33 = 'pathlib'
    var_34 = '/project/main.py'
    var_35 = str(var_30)
    assert var_35 == '/project/main.py:7 import pathlib'
    var_36 = 8
    var_37 = True
    var_38 = 'libc'
    var_39 = 'stdio'
    var_40 = 'cstdio'
    var_41 = True
    var_42 = 'module.py'
    var_43 = str(var_30)
    assert var_43 == 'module.py:8 indented from libc cimport stdio as cstdio'



# Parsed testcases at query #86
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



# Parsed testcases at query #87
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
    var_9 = 'system'
    var_10 = '/another/test.py'
    var_11 = 3
    var_12 = 'math'
    var_13 = 'sqrt'
    var_14 = '/math/test.py'
    var_15 = 7
    var_16 = 'cython'
    var_17 = 'view'
    var_18 = '/cython/test.py'
    var_19 = 'collections'
    var_20 = 'defaultdict'
    var_21 = 'dd'
    var_22 = module_0.Import()
    var_23 = str(var_22)
    assert var_23 == ':1 from collections import defaultdict as dd'



# Parsed testcases at query #88
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



# Parsed testcases at query #89
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import numpy as np\n'
    var_2 = 'from os import path\n'
    var_3 = 'from os import path as p\n'
    var_4 = 'cimport numpy\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import (\n    path,\n    environ\n)\n'
    var_7 = '    import os\n'
    var_8 = 'import os  # This is a comment\n'
    var_9 = 'import os; import sys\n'
    var_10 = 'import os as os\n'
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'import os\ndef foo():\n    import sys\n'



# Parsed testcases at query #90
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
    var_28 = 'libc'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':6 cimport libc'
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
    var_42 = str(var_30)
    assert var_42 == '/tmp/test.py:8 indented from django cimport models as dm'



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'numpy'
    var_4 = 'np'
    var_5 = 'module'
    var_6 = True
    var_7 = 'cython_module'
    var_8 = True
    var_9 = 'cm'
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'dd'
    var_13 = 'libc'
    var_14 = 'stdio'
    var_15 = True
    var_16 = True
    var_17 = 'cstdio'



# Parsed testcases at query #92
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'operating_system'
    var_6 = module_0.Import()
    var_7 = var_6.statement()
    assert var_7 == 'import os as operating_system'
    var_8 = 'numpy'
    var_9 = True
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'cimport numpy'
    var_12 = 'np'
    var_13 = True
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'cimport numpy as np'
    var_16 = 'path'
    var_17 = module_0.Import()
    var_18 = var_17.statement()
    assert var_18 == 'from os import path'
    var_19 = 'array'
    var_20 = True
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from numpy cimport array'
    var_23 = 'p'
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from os import path as p'
    var_26 = 'arr'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from numpy cimport array as arr'



# Parsed testcases at query #93
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
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 import numpy as np'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 from collections import defaultdict'
    var_20 = 5
    var_21 = True
    var_22 = 'typing'
    var_23 = 'List'
    var_24 = 'T'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 indented from typing import List as T'
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
    var_38 = 'unittest'
    var_39 = str(var_30)
    assert var_39 == '/tmp/test.py:8 indented import unittest'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_30 = 'from collections import (\n'
    var_31 = '    defaultdict,\n'
    var_32 = '    OrderedDict\n'
    var_33 = ')\n'
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = '    import os\n'
    var_39 = [var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
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
    var_60 = 'def foo():\n'
    var_61 = '    import sys\n'
    var_62 = [var_0, var_60, var_61]
    var_63 = module_0.imports(var_62, top_only=var_53)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = '/some/path/file.py'
    var_67 = [var_0]
    var_68 = len(var_64)
    assert var_68 == 1



# Parsed testcases at query #2
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
    var_21 = 'cfunc'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from cython cimport cfunc'
    var_25 = 'p'
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from os import path as p'
    var_28 = 'cf'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #3
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = 1
    var_3 = False
    var_4 = 'os'
    var_5 = module_0.imports(var_1)
    var_6 = list(var_5)
    var_7 = 'import numpy as np\n'
    var_8 = [var_7]
    var_9 = 'numpy'
    var_10 = 'np'
    var_11 = module_0.imports(var_8)
    var_12 = list(var_11)
    var_13 = 'from collections import defaultdict\n'
    var_14 = [var_13]
    var_15 = 'collections'
    var_16 = 'defaultdict'
    var_17 = module_0.imports(var_14)
    var_18 = list(var_17)
    var_19 = 'from pathlib import Path as P\n'
    var_20 = [var_19]
    var_21 = 'pathlib'
    var_22 = 'Path'
    var_23 = 'P'
    var_24 = module_0.imports(var_20)
    var_25 = list(var_24)
    var_26 = 'cimport numpy\n'
    var_27 = [var_26]
    var_28 = True
    var_29 = module_0.imports(var_27)
    var_30 = list(var_29)
    var_31 = 'import sys, os\n'
    var_32 = [var_31]
    var_33 = 'sys'
    var_34 = module_0.imports(var_32)
    var_35 = list(var_34)
    var_36 = 'from typing import (\n'
    var_37 = '    List,\n'
    var_38 = '    Dict,\n'
    var_39 = ')\n'
    var_40 = [var_36, var_37, var_38, var_39]
    var_41 = 'typing'
    var_42 = 'List'
    var_43 = 'Dict'
    var_44 = module_0.imports(var_40)
    var_45 = list(var_44)
    var_46 = '    import sys\n'
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.imports(var_47)
    var_50 = list(var_49)
    var_51 = 'import os  # Operating system interfaces\n'
    var_52 = [var_51]
    var_53 = module_0.imports(var_52)
    var_54 = list(var_53)
    var_55 = True
    var_56 = module_1.Config()
    var_57 = 'import numpy as numpy\n'
    var_58 = [var_57]
    var_59 = module_0.imports(var_58, var_56)
    var_60 = list(var_59)
    var_61 = 'def foo():\n'
    var_62 = [var_0, var_61, var_46]
    var_63 = True
    var_64 = module_0.imports(var_62, top_only=var_63)
    var_65 = list(var_64)



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
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == ':2 import numpy as np'
    var_10 = 3
    var_11 = 'sys'
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 from sys import path'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict as dd'
    var_21 = 5
    var_22 = True
    var_23 = 'typing'
    var_24 = 'List'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 indented from typing import List'
    var_27 = 6
    var_28 = 'libc'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':6 cimport libc'
    var_32 = 7
    var_33 = 'pathlib'
    var_34 = '/tmp/test.py'
    var_35 = str(var_30)
    assert var_35 == '/tmp/test.py:7 import pathlib'



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = True
    var_2 = 'import numpy as np\n'
    var_3 = 'from collections import defaultdict\n'
    var_4 = 'from pathlib import Path as P\n'
    var_5 = 'cimport numpy\n'
    var_6 = 'import os, sys\n'
    var_7 = 'from collections import (\n    defaultdict,\n    OrderedDict\n)\n'
    var_8 = '    import os\n'
    var_9 = 'import os  # some comment\n'
    var_10 = '/tmp/test.py'
    var_11 = 'import os\n'
    var_12 = 'import os\n\ndef foo():\n    pass\n'
    var_13 = module_0.Config()
    var_14 = 'import numpy as numpy\n'
    var_15 = 'TYPE CHECKING'
    var_16 = 'typing'
    var_17 = {var_15: var_16}
    var_18 = module_0.Config()
    var_19 = '# TYPE CHECKING\nimport typing\n'
    var_20 = "'''\nimport os\n'''\n"
    var_21 = 'def foo():\n    yield\n    import os\n'
    var_22 = 'raise ValueError\nimport os\n'
    var_23 = 'import os \\\n    , sys\n'
    var_24 = 'import os; import sys\n'



# Parsed testcases at query #6
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'ospath'
    var_5 = '/path/to/file.py'
    var_6 = 2
    var_7 = True
    var_8 = 'sys'
    var_9 = None
    var_10 = True
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':2 indented cimport sys'
    var_13 = 3
    var_14 = 'math'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':3 import math'
    var_17 = 4
    var_18 = True
    var_19 = 'datetime'
    var_20 = 'date'
    var_21 = module_0.Import()
    var_22 = str(var_21)
    assert var_22 == ':4 indented from datetime import date'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'module'
    var_3 = 'attr'
    var_4 = 'alias'
    var_5 = True
    var_6 = 'file.py'
    var_7 = 2
    var_8 = True
    var_9 = None
    var_10 = 3
    var_11 = 'test.py'
    var_12 = 4



# Parsed testcases at query #9
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
    var_30 = 'from typing import (\n    List,\n    Dict,\n)'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import sys'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = '/test/file.py'
    var_41 = [var_0]
    var_42 = len(var_38)
    assert var_42 == 1
    var_43 = 'def func():'
    var_44 = [var_0, var_43, var_35]
    var_45 = True
    var_46 = module_0.imports(var_44, top_only=var_45)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = module_1.Config()
    var_50 = 'import numpy as numpy'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51, var_49)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 1



# Parsed testcases at query #10
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
    var_30 = '    import os\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from os import (\n    path,\n    environ\n)\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '# This is a comment\nimport os # inline comment\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = '"""\nThis is a docstring\n"""'
    var_46 = [var_45, var_0]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = 'def foo():\n'
    var_51 = '    import sys\n'
    var_52 = [var_0, var_50, var_51]
    var_53 = True
    var_54 = module_0.imports(var_52, top_only=var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/test/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = module_1.Config()
    var_61 = 'import os as os\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62, var_60)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = '    pass\n'
    var_67 = [var_0, var_50, var_66]
    var_68 = module_0.imports(var_67, top_only=var_53)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1



# Parsed testcases at query #11
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
    var_20 = 'p'
    var_21 = module_0.Import()
    var_22 = var_21.statement()
    assert var_22 == 'from os import path as p'
    var_23 = 'cfunc'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport cfunc'
    var_27 = 'cf'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #13
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
    var_21 = 'from typing import List, Dict, Set\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = 'typing'
    var_27 = 'cimport numpy\n'
    var_28 = [var_27]
    var_29 = module_0.imports(var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = 'from collections import (\n'
    var_33 = '    defaultdict,\n'
    var_34 = '    Counter\n'
    var_35 = ')\n'
    var_36 = [var_32, var_33, var_34, var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = 'collections'
    var_41 = 'import os  # Operating system interfaces\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '# This is a comment\n'
    var_47 = '\n'
    var_48 = [var_46, var_47, var_1]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = '    import os\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = True
    var_58 = module_1.Config()
    var_59 = 'import numpy as numpy\n'
    var_60 = [var_59]
    var_61 = module_0.imports(var_60, var_58)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = 'def foo():\n'
    var_65 = '    import sys\n'
    var_66 = [var_0, var_64, var_65]
    var_67 = module_0.imports(var_66, top_only=var_57)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1
    var_70 = '/path/to/file.py'
    var_71 = [var_0]
    var_72 = len(var_68)
    assert var_72 == 1



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'libc'
    var_22 = 'stdio'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from libc cimport stdio'
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'dd'
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from collections import defaultdict as dd'



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
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'path'
    var_18 = 'import numpy as np\n'
    var_19 = [var_18]
    var_20 = module_0.imports(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 'numpy'
    var_24 = 'np'
    var_25 = 'from os import path as p\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'p'
    var_31 = 'cimport numpy\n'
    var_32 = [var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = True
    var_37 = 'from os import (\n'
    var_38 = '    path,\n'
    var_39 = '    environ\n'
    var_40 = ')\n'
    var_41 = [var_37, var_38, var_39, var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = 3
    var_46 = 'environ'
    var_47 = '    import os\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = True
    var_53 = True
    var_54 = module_1.Config()
    var_55 = 'import os as os\n'
    var_56 = [var_55]
    var_57 = module_0.imports(var_56, var_54)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = '/test/path'
    var_61 = [var_0]
    var_62 = len(var_58)
    assert var_62 == 1
    var_63 = 'def func():\n'
    var_64 = '    import sys\n'
    var_65 = [var_0, var_63, var_64]
    var_66 = True
    var_67 = module_0.imports(var_65, top_only=var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1



# Parsed testcases at query #17
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
    var_17 = 'os.path'
    var_18 = 'join'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from os.path import join'
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



# Parsed testcases at query #18
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
    var_20 = 'cfunc'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport cfunc'
    var_24 = 'cf'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #19
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
    var_7 = 1
    var_8 = False
    var_9 = 'os'
    var_10 = 2
    var_11 = 'sys'
    var_12 = 'import numpy as np\n'
    var_13 = [var_12]
    var_14 = iter(var_13)
    var_15 = module_0.imports(var_14)
    var_16 = list(var_15)
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = 'numpy'
    var_19 = 'np'
    var_20 = 'from collections import defaultdict\n'
    var_21 = [var_20]
    var_22 = iter(var_21)
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 1
    var_26 = 'collections'
    var_27 = 'defaultdict'
    var_28 = 'from pathlib import Path as P\n'
    var_29 = [var_28]
    var_30 = iter(var_29)
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = 'pathlib'
    var_35 = 'Path'
    var_36 = 'P'
    var_37 = 'cimport numpy\n'
    var_38 = [var_37]
    var_39 = iter(var_38)
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = True
    var_44 = 'from collections import (\n'
    var_45 = '    defaultdict,\n'
    var_46 = '    Counter\n'
    var_47 = ')\n'
    var_48 = [var_44, var_45, var_46, var_47]
    var_49 = iter(var_48)
    var_50 = module_0.imports(var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = 3
    var_54 = 'Counter'
    var_55 = '    import os\n'
    var_56 = [var_55]
    var_57 = iter(var_56)
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = True
    var_62 = True
    var_63 = module_1.Config()
    var_64 = 'import numpy as numpy\n'
    var_65 = [var_64]
    var_66 = iter(var_65)
    var_67 = module_0.imports(var_66, var_63)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1
    var_70 = '# This is a comment\n'
    var_71 = 'import os  # inline comment\n'
    var_72 = [var_70, var_71]
    var_73 = iter(var_72)
    var_74 = module_0.imports(var_73)
    var_75 = list(var_74)
    var_76 = len(var_75)
    assert var_76 == 1
    var_77 = 'def foo():\n'
    var_78 = '    import sys\n'
    var_79 = [var_0, var_77, var_78]
    var_80 = iter(var_79)
    var_81 = True
    var_82 = module_0.imports(var_80, top_only=var_81)
    var_83 = list(var_82)
    var_84 = len(var_83)
    assert var_84 == 1
    var_85 = '/test/file.py'
    var_86 = [var_0]
    var_87 = iter(var_86)
    var_88 = len(var_83)
    assert var_88 == 1



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'module'
    var_3 = 'attribute'
    var_4 = 'alias'
    var_5 = '/path/to/file'
    var_6 = 5
    var_7 = False
    var_8 = None
    var_9 = '/test'
    var_10 = 3
    var_11 = '/another/path'
    var_12 = 7



# Parsed testcases at query #22
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
    var_10 = 'from sys import argv\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pandas import DataFrame as DF\n'
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
    var_30 = 'from os import path, environ\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '    import os\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'import os  # Some comment\n'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = 'import os; import sys\n'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = '/some/path/file.py'
    var_56 = [var_0]
    var_57 = len(var_53)
    assert var_57 == 1
    var_58 = True
    var_59 = module_1.Config()
    var_60 = 'import numpy as numpy\n'
    var_61 = [var_60]
    var_62 = module_0.imports(var_61, var_59)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = 'def foo():\n'
    var_66 = '    import sys\n'
    var_67 = [var_0, var_65, var_66]
    var_68 = module_0.imports(var_67, top_only=var_58)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = '"""Module docstring"""'
    var_72 = [var_71, var_0]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = 'from os import \\\n    path\n'
    var_77 = [var_76]
    var_78 = module_0.imports(var_77)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 1
    var_81 = '    yield\n'
    var_82 = [var_65, var_81, var_40]
    var_83 = module_0.imports(var_82)
    var_84 = list(var_83)
    var_85 = len(var_84)
    assert var_85 == 1
    var_86 = 'raise ValueError\n'
    var_87 = [var_86, var_0]
    var_88 = module_0.imports(var_87)
    var_89 = list(var_88)
    var_90 = len(var_89)
    assert var_90 == 1



# Parsed testcases at query #23
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
    var_11 = 'os.path'
    var_12 = 'join'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 from os.path import join'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict as dd'
    var_21 = 5
    var_22 = True
    var_23 = 'sys'
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 indented import sys'
    var_26 = 6
    var_27 = 'cython'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = str(var_29)
    assert var_30 == ':6 cimport cython'
    var_31 = 7
    var_32 = 'pathlib'
    var_33 = '/tmp/test.py'
    var_34 = str(var_29)
    assert var_34 == '/tmp/test.py:7 import pathlib'



# Parsed testcases at query #24
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
    var_23 = 'func'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport func'
    var_27 = 'f'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport func as f'



# Parsed testcases at query #25
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
    var_8 = 'sys'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import sys'
    var_11 = 3
    var_12 = 'numpy'
    var_13 = 'np'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 import numpy as np'
    var_16 = 4
    var_17 = True
    var_18 = 'pandas'
    var_19 = 'pd'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 indented import pandas as pd'
    var_22 = 5
    var_23 = 'collections'
    var_24 = 'defaultdict'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 from collections import defaultdict'
    var_27 = 6
    var_28 = 'dd'
    var_29 = module_0.Import()
    var_30 = str(var_29)
    assert var_30 == ':6 from collections import defaultdict as dd'
    var_31 = 7
    var_32 = 'cython'
    var_33 = True
    var_34 = module_0.Import()
    var_35 = str(var_34)
    assert var_35 == ':7 cimport cython'
    var_36 = 8
    var_37 = 'cy'
    var_38 = True
    var_39 = module_0.Import()
    var_40 = str(var_39)
    assert var_40 == ':8 cimport cython as cy'
    var_41 = 9
    var_42 = 'test.py'
    var_43 = str(var_39)
    assert var_43 == 'test.py:9 import os'
    var_44 = 10
    var_45 = True
    var_46 = str(var_39)
    assert var_46 == 'test.py:10 indented from collections import defaultdict'



# Parsed testcases at query #26
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
    var_34 = '    import os\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = module_1.Config()
    var_41 = 'import numpy as numpy\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42, var_40)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = 'def foo():\n'
    var_47 = '    import sys\n'
    var_48 = [var_0, var_46, var_47]
    var_49 = module_0.imports(var_48, top_only=var_39)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = '/tmp/test.py'
    var_53 = [var_0]
    var_54 = len(var_50)
    assert var_54 == 1
    var_55 = '# This is a comment\n'
    var_56 = [var_55, var_0]
    var_57 = module_0.imports(var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = 'numpy'
    var_5 = True
    var_6 = True
    var_7 = 'np'
    var_8 = 'path'
    var_9 = True
    var_10 = 'p'
    var_11 = True



# Parsed testcases at query #28
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = var_3.statement()
    assert var_4 == 'import os'
    var_5 = 'operating_system'
    var_6 = module_0.Import()
    var_7 = var_6.statement()
    assert var_7 == 'import os as operating_system'
    var_8 = 'numpy'
    var_9 = True
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'cimport numpy'
    var_12 = True
    var_13 = 'np'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'cimport numpy as np'
    var_16 = 'path'
    var_17 = module_0.Import()
    var_18 = var_17.statement()
    assert var_18 == 'from os import path'
    var_19 = True
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'from os cimport path'
    var_22 = 'p'
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from os import path as p'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from os cimport path as p'



# Parsed testcases at query #29
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
    var_7 = 1
    var_8 = False
    var_9 = 'os'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = 2
    var_13 = 'sys'
    var_14 = module_0.Import()
    var_15 = 'from os import path\n'
    var_16 = 'from sys import argv\n'
    var_17 = [var_15, var_16]
    var_18 = iter(var_17)
    var_19 = module_0.imports(var_18)
    var_20 = list(var_19)
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = 'path'
    var_23 = module_0.Import()
    var_24 = 'argv'
    var_25 = module_0.Import()
    var_26 = 'import numpy as np\n'
    var_27 = 'from pandas import DataFrame as df\n'
    var_28 = [var_26, var_27]
    var_29 = iter(var_28)
    var_30 = module_0.imports(var_29)
    var_31 = list(var_30)
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'numpy'
    var_34 = 'np'
    var_35 = module_0.Import()
    var_36 = 'pandas'
    var_37 = 'DataFrame'
    var_38 = 'df'
    var_39 = module_0.Import()
    var_40 = 'cimport numpy\n'
    var_41 = 'from cython cimport int\n'
    var_42 = [var_40, var_41]
    var_43 = iter(var_42)
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = True
    var_48 = module_0.Import()
    var_49 = 'cython'
    var_50 = 'int'
    var_51 = True
    var_52 = module_0.Import()
    var_53 = 'from os import (\n'
    var_54 = '    path,\n'
    var_55 = '    environ\n'
    var_56 = ')\n'
    var_57 = [var_53, var_54, var_55, var_56]
    var_58 = iter(var_57)
    var_59 = module_0.imports(var_58)
    var_60 = list(var_59)
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = module_0.Import()
    var_63 = 3
    var_64 = 'environ'
    var_65 = module_0.Import()
    var_66 = '    import os\n'
    var_67 = [var_66, var_1]
    var_68 = iter(var_67)
    var_69 = module_0.imports(var_68)
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = True
    var_73 = module_0.Import()
    var_74 = module_0.Import()
    var_75 = '/test/file.py'
    var_76 = [var_0]
    var_77 = iter(var_76)
    var_78 = len(var_70)
    assert var_78 == 1
    var_79 = 'def foo():\n'
    var_80 = '    import sys\n'
    var_81 = [var_0, var_79, var_80]
    var_82 = iter(var_81)
    var_83 = True
    var_84 = module_0.imports(var_82, top_only=var_83)
    var_85 = list(var_84)
    var_86 = len(var_85)
    assert var_86 == 1
    var_87 = module_0.Import()
    var_88 = '# This is a comment\n'
    var_89 = 'import os  # inline comment\n'
    var_90 = [var_88, var_89]
    var_91 = iter(var_90)
    var_92 = module_0.imports(var_91)
    var_93 = list(var_92)
    var_94 = len(var_93)
    assert var_94 == 1
    var_95 = module_0.Import()
    var_96 = 'import os; import sys\n'
    var_97 = [var_96]
    var_98 = iter(var_97)
    var_99 = module_0.imports(var_98)
    var_100 = list(var_99)
    var_101 = len(var_100)
    assert var_101 == 2
    var_102 = module_0.Import()
    var_103 = module_0.Import()
    var_104 = True
    var_105 = module_1.Config()
    var_106 = 'import os as os\n'
    var_107 = 'from sys import argv as argv\n'
    var_108 = [var_106, var_107]
    var_109 = iter(var_108)
    var_110 = module_0.imports(var_109, var_105)
    var_111 = list(var_110)
    var_112 = len(var_111)
    assert var_112 == 2
    var_113 = module_0.Import()
    var_114 = module_0.Import()



# Parsed testcases at query #30
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
    var_15 = 'import sys, os\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = True
    var_31 = module_1.Config()
    var_32 = 'import numpy as numpy\n'
    var_33 = [var_32]
    var_34 = module_0.imports(var_33, var_31)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = '    import os\n'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 1
    var_42 = '# This is a comment\nimport os\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 'x = 5\nimport os\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'import os\ndef foo():\n    import sys\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53, top_only=var_30)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/test/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = None
    var_4 = module_0.Import()
    var_5 = str(var_4)
    assert var_5 == '1 import os'
    var_6 = 2
    var_7 = True
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == '2 indented import numpy as np'
    var_12 = 3
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == '3 from collections import defaultdict'
    var_17 = 4
    var_18 = True
    var_19 = 'dd'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == '4 indented from collections import defaultdict as dd'
    var_22 = 5
    var_23 = 'cython'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == '5 cimport cython'
    var_27 = 6
    var_28 = 'sys'
    var_29 = '/path/to/file.py'
    var_30 = str(var_25)
    assert var_30 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'module'
    var_3 = None
    var_4 = 'alias'
    var_5 = 'attribute'
    var_6 = True
    var_7 = True
    var_8 = True
    var_9 = True



# Parsed testcases at query #34
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
    var_21 = 'from typing import List, Dict, Set\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = 'cimport numpy\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'from libc cimport malloc\n'
    var_32 = [var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = 'from collections import (\n'
    var_37 = '    defaultdict,\n'
    var_38 = '    OrderedDict,\n'
    var_39 = ')\n'
    var_40 = [var_36, var_37, var_38, var_39]
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
    var_49 = 'import os  # Operating system interfaces\n'
    var_50 = [var_49]
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = True
    var_55 = module_1.Config()
    var_56 = 'import os as os\n'
    var_57 = [var_56]
    var_58 = module_0.imports(var_57, var_55)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = '/path/to/file.py'
    var_62 = [var_1]
    var_63 = len(var_59)
    assert var_63 == 1
    var_64 = 'def function():\n'
    var_65 = '    import sys\n'
    var_66 = [var_0, var_64, var_65]
    var_67 = module_0.imports(var_66, top_only=var_54)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 1



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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
    var_15 = 'from pandas import DataFrame as df'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import os, sys'
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
    var_30 = 'from libc cimport printf'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 1
    var_35 = 'from os import (\n    path,\n    sys\n)'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = '    import os'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = 'import os # This is a comment'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = True
    var_51 = module_1.Config()
    var_52 = 'import numpy as numpy'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53, var_51)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = 'def foo():'
    var_58 = '    import sys'
    var_59 = [var_0, var_57, var_58]
    var_60 = module_0.imports(var_59, top_only=var_50)
    var_61 = list(var_60)
    var_62 = len(var_61)
    assert var_62 == 1
    var_63 = '/test/path.py'
    var_64 = [var_0]
    var_65 = len(var_61)
    assert var_65 == 1



# Parsed testcases at query #37
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
    var_20 = 'cimport numpy'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import os, sys'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from typing import (\n    List,\n    Dict,\n)'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import os'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = '/test.py'
    var_41 = [var_0]
    var_42 = len(var_38)
    assert var_42 == 1
    var_43 = 'import os  # This is a comment'
    var_44 = [var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = 'import os; import sys'
    var_49 = [var_48]
    var_50 = module_0.imports(var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = 'import numpy as numpy'
    var_54 = [var_53]
    var_55 = True
    var_56 = module_1.Config()
    var_57 = module_0.imports(var_54, var_56)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = 'def func():'
    var_61 = '    import sys'
    var_62 = [var_0, var_60, var_61]
    var_63 = module_0.imports(var_62, top_only=var_55)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'from typing import \\\n    List'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = 'import(\nos\n)'
    var_72 = [var_71]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1
    var_76 = 'yield'
    var_77 = [var_76, var_0]
    var_78 = module_0.imports(var_77)
    var_79 = list(var_78)
    var_80 = len(var_79)
    assert var_80 == 1
    var_81 = 'raise ValueError'
    var_82 = [var_81, var_0]
    var_83 = module_0.imports(var_82)
    var_84 = list(var_83)
    var_85 = len(var_84)
    assert var_85 == 1



# Parsed testcases at query #38
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
    var_9 = 'exit'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':5 from sys import exit'
    var_13 = 'math'
    var_14 = 'script.py'
    var_15 = str(var_11)
    assert var_15 == 'script.py:1 cimport math'
    var_16 = 3
    var_17 = 'numpy'
    var_18 = 'analysis.py'
    var_19 = str(var_11)
    assert var_19 == 'analysis.py:3 indented import numpy as numpy'



# Parsed testcases at query #39
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
    var_46 = 'x = 1\n'
    var_47 = 'y = 2\n'
    var_48 = [var_46, var_0, var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'def foo():\n'
    var_53 = '    import sys\n'
    var_54 = [var_0, var_52, var_53]
    var_55 = module_0.imports(var_54, top_only=var_34)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = '/test.py'
    var_59 = [var_0]
    var_60 = len(var_56)
    assert var_60 == 1



# Parsed testcases at query #40
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
    var_20 = 'cimport numpy'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'import os, sys'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 2
    var_30 = 'from collections import (\n    defaultdict,\n    Counter\n)'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = '    import os'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = '/test/path'
    var_41 = [var_0]
    var_42 = len(var_38)
    assert var_42 == 1
    var_43 = True
    var_44 = module_1.Config()
    var_45 = 'import numpy as numpy'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46, var_44)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = 'def func():'
    var_51 = '    import sys'
    var_52 = [var_0, var_50, var_51]
    var_53 = module_0.imports(var_52, top_only=var_43)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1



# Parsed testcases at query #41
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
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict'
    var_21 = 5
    var_22 = 'typing'
    var_23 = 'List'
    var_24 = 'list'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 from typing import List as list'
    var_27 = 6
    var_28 = 'cython'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':6 cimport cython'
    var_32 = 7
    var_33 = 'pathlib'
    var_34 = '/test.py'
    var_35 = str(var_30)
    assert var_35 == '/test.py:7 import pathlib'



# Parsed testcases at query #42
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
    var_21 = 'cimport numpy as np\n'
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
    var_46 = '# This is a comment\n'
    var_47 = 'import os  # inline comment\n'
    var_48 = [var_46, var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'x = 1\n'
    var_53 = 'def foo():\n'
    var_54 = '    pass\n'
    var_55 = [var_52, var_0, var_53, var_54]
    var_56 = module_0.imports(var_55)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = '    import sys\n'
    var_60 = [var_0, var_53, var_59]
    var_61 = module_0.imports(var_60, top_only=var_34)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = '/test/path.py'
    var_65 = [var_0]
    var_66 = len(var_62)
    assert var_66 == 1



# Parsed testcases at query #43
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
    var_9 = 's'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':5 cimport sys as s'
    var_13 = 15
    var_14 = 'collections'
    var_15 = 'defaultdict'
    var_16 = 'example.py'
    var_17 = str(var_11)
    assert var_17 == 'example.py:15 indented from collections import defaultdict'
    var_18 = 20
    var_19 = 'math'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':20 cimport math'
    var_22 = 25
    var_23 = 'json'
    var_24 = 'loads'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':25 indented from json import loads'



# Parsed testcases at query #44
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import numpy as np\n'
    var_2 = 'from collections import defaultdict\n'
    var_3 = 'from pathlib import Path as P\n'
    var_4 = 'import sys, os\n'
    var_5 = 'cimport numpy\n'
    var_6 = '    import os\n'
    var_7 = 'from collections import (\n    defaultdict,\n    Counter\n)\n'
    var_8 = 'import os  # Operating system\n'
    var_9 = '\nimport os\n'
    var_10 = 'import sys\nimport os\n'
    var_11 = '/test/file.py'
    var_12 = 'import os\n'
    var_13 = 'import os\ndef foo():\n    import sys\n'
    var_14 = True
    var_15 = 'import numpy as numpy\n'
    var_16 = module_0.Config()
    var_17 = False
    var_18 = 'os'
    var_19 = None
    var_20 = 'numpy'
    var_21 = 'np'
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = 'dd'
    var_25 = '/test.py'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = 'module'
    var_3 = 'attribute'
    var_4 = 'alias'
    var_5 = True
    var_6 = 'file.py'
    var_7 = 2
    var_8 = False
    var_9 = None
    var_10 = 3
    var_11 = True
    var_12 = 'test.py'
    var_13 = 4
    var_14 = True
    var_15 = 'path/to/file.py'



# Parsed testcases at query #46
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
    var_8 = 'sys'
    var_9 = module_0.Import()
    var_10 = str(var_9)
    assert var_10 == ':2 indented import sys'
    var_11 = 3
    var_12 = 'pathlib'
    var_13 = 'test.py'
    var_14 = str(var_9)
    assert var_14 == 'test.py:3 import pathlib'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 from collections import defaultdict'
    var_20 = 5
    var_21 = 'numpy'
    var_22 = 'np'
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':5 import numpy as np'
    var_25 = 6
    var_26 = 'libc'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 cimport libc'
    var_30 = 7
    var_31 = True
    var_32 = 'OrderedDict'
    var_33 = 'OD'
    var_34 = True
    var_35 = 'example.py'
    var_36 = str(var_28)
    assert var_36 == 'example.py:7 indented from collections cimport OrderedDict as OD'



# Parsed testcases at query #47
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
    var_18 = 'sys'
    var_19 = 'path'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 from sys import path'
    var_22 = 5
    var_23 = True
    var_24 = 'stdio'
    var_25 = 'cstdio'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = str(var_27)
    assert var_28 == ':5 indented from libc cimport stdio as cstdio'
    var_29 = 6
    var_30 = '/path/to/file.py'
    var_31 = str(var_27)
    assert var_31 == '/path/to/file.py:6 import os'



# Parsed testcases at query #48
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
    var_21 = 'libc'
    var_22 = 'stdio'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from libc cimport stdio'
    var_26 = 'p'
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'from os import path as p'
    var_29 = 's'
    var_30 = True
    var_31 = module_0.Import()
    var_32 = var_31.statement()
    assert var_32 == 'from libc cimport stdio as s'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = 'module'
    var_3 = 'attribute'
    var_4 = 'alias'
    var_5 = True
    var_6 = '/path/to/file'
    var_7 = 2
    var_8 = False
    var_9 = None
    var_10 = 3
    var_11 = 4
    var_12 = True
    var_13 = True



# Parsed testcases at query #51
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
    var_12 = 'collections'
    var_13 = 'defaultdict'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 from collections import defaultdict'
    var_16 = 4
    var_17 = 'dd'
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 from collections import defaultdict as dd'
    var_20 = 5
    var_21 = 'cython'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':5 cimport cython'
    var_25 = 6
    var_26 = True
    var_27 = 'sys'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 indented import sys'
    var_30 = 7
    var_31 = '/path/to/file.py'
    var_32 = str(var_28)
    assert var_32 == '/path/to/file.py:7 import os'
    var_33 = 8
    var_34 = 'libc'
    var_35 = 'stdio'
    var_36 = 'cstdio'
    var_37 = True
    var_38 = module_0.Import()
    var_39 = str(var_38)
    assert var_39 == ':8 from libc cimport stdio as cstdio'



# Parsed testcases at query #52
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
    var_34 = 'import os  # Operating system interfaces\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = module_1.Config()
    var_41 = 'import numpy as numpy\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42, var_40)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '    import os\n'
    var_47 = [var_46]
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = 'def foo():\n'
    var_52 = '    import sys\n'
    var_53 = [var_0, var_51, var_52]
    var_54 = module_0.imports(var_53, top_only=var_39)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = '/path/to/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1



# Parsed testcases at query #53
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
    var_15 = 'import os, sys\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = '    import os\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'from collections import (\n    OrderedDict,\n    defaultdict\n)\n'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'import os  # Operating system\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = True
    var_41 = module_1.Config()
    var_42 = 'import os as os\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43, var_41)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = '/test/file.py'
    var_48 = [var_0]
    var_49 = len(var_45)
    assert var_49 == 1
    var_50 = 'def function():\n'
    var_51 = '    import sys\n'
    var_52 = [var_0, var_50, var_51]
    var_53 = module_0.imports(var_52, top_only=var_40)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 1



# Parsed testcases at query #54
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
    var_11 = True
    var_12 = 'sys'
    var_13 = 'path'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 indented from sys import path'
    var_16 = 4
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'dd'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 from collections import defaultdict as dd'
    var_22 = 5
    var_23 = 'libc'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 cimport libc'
    var_27 = 6
    var_28 = '/path/to/file.py'
    var_29 = str(var_25)
    assert var_29 == '/path/to/file.py:6 import os'



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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
    var_11 = 'sys'
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 from sys import path'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 from collections import defaultdict as dd'
    var_21 = 5
    var_22 = True
    var_23 = 'typing'
    var_24 = 'List'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':5 indented from typing import List'
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



# Parsed testcases at query #57
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
    var_34 = 'import os  # Operating system interfaces\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = module_1.Config()
    var_41 = 'import numpy as numpy\n'
    var_42 = [var_41]
    var_43 = module_0.imports(var_42, var_40)
    var_44 = list(var_43)
    var_45 = len(var_44)
    assert var_45 == 1
    var_46 = '    import sys\n'
    var_47 = [var_46]
    var_48 = module_0.imports(var_47)
    var_49 = list(var_48)
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = '/test/file.py'
    var_52 = [var_0]
    var_53 = len(var_49)
    assert var_53 == 1
    var_54 = 'def function():\n'
    var_55 = '    pass\n'
    var_56 = [var_0, var_54, var_55]
    var_57 = module_0.imports(var_56, top_only=var_39)
    var_58 = list(var_57)
    var_59 = len(var_58)
    assert var_59 == 1
    var_60 = "'''Module docstring'''\n"
    var_61 = [var_60, var_0]
    var_62 = module_0.imports(var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = 'import os; import sys\n'
    var_66 = [var_65]
    var_67 = module_0.imports(var_66)
    var_68 = list(var_67)
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = 'from typing import \\\n'
    var_71 = '    List\n'
    var_72 = [var_70, var_71]
    var_73 = module_0.imports(var_72)
    var_74 = list(var_73)
    var_75 = len(var_74)
    assert var_75 == 1



# Parsed testcases at query #58
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
    var_9 = 'libc'
    var_10 = True
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'cimport libc'
    var_13 = 'c'
    var_14 = True
    var_15 = module_0.Import()
    var_16 = var_15.statement()
    assert var_16 == 'cimport libc as c'
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'collections'
    var_22 = 'defaultdict'
    var_23 = 'dd'
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from collections import defaultdict as dd'
    var_26 = 'stdio'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from libc cimport stdio'
    var_30 = 'cstdio'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = var_32.statement()
    assert var_33 == 'from libc cimport stdio as cstdio'



# Parsed testcases at query #59
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
    var_12 = 'import numpy as np\n'
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'numpy'
    var_18 = 'np'
    var_19 = 'from collections import defaultdict\n'
    var_20 = [var_19]
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'collections'
    var_25 = 'defaultdict'
    var_26 = 'from pathlib import Path as P\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'pathlib'
    var_32 = 'Path'
    var_33 = 'P'
    var_34 = 'cimport numpy\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = 'import os, sys\n'
    var_41 = [var_40]
    var_42 = module_0.imports(var_41)
    var_43 = list(var_42)
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = '    import os\n'
    var_46 = [var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = True
    var_51 = '/test/path'
    var_52 = [var_0]
    var_53 = len(var_48)
    assert var_53 == 1
    var_54 = 'def func():\n'
    var_55 = '    import sys\n'
    var_56 = [var_0, var_54, var_55]
    var_57 = True
    var_58 = module_0.imports(var_56, top_only=var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = 'import os  # comment\n'
    var_62 = [var_61]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = len(var_64)
    assert var_65 == 1
    var_66 = 'from collections import (\n'
    var_67 = '    defaultdict,\n'
    var_68 = '    OrderedDict\n'
    var_69 = ')\n'
    var_70 = [var_66, var_67, var_68, var_69]
    var_71 = module_0.imports(var_70)
    var_72 = list(var_71)
    var_73 = len(var_72)
    assert var_73 == 2
    var_74 = 'OrderedDict'
    var_75 = True
    var_76 = module_1.Config()
    var_77 = 'import os as os\n'
    var_78 = [var_77]
    var_79 = module_0.imports(var_78, var_76)
    var_80 = list(var_79)
    var_81 = len(var_80)
    assert var_81 == 1



# Parsed testcases at query #60
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
    var_9 = 'sys'
    var_10 = 'path'
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'from sys import path'
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
    var_26 = 'cdef'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from cython cimport cdef'



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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



# Parsed testcases at query #63
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
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':4 from collections import defaultdict'
    var_22 = 5
    var_23 = True
    var_24 = 'libc'
    var_25 = 'stdio'
    var_26 = 'libc_stdio'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':5 indented from libc cimport stdio as libc_stdio'
    var_30 = 6
    var_31 = 'sys'
    var_32 = '/path/to/file.py'
    var_33 = str(var_28)
    assert var_33 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #64
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
    var_12 = 'import numpy as np\n'
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'numpy'
    var_18 = 'np'
    var_19 = 'from collections import defaultdict\n'
    var_20 = [var_19]
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1
    var_24 = 'collections'
    var_25 = 'defaultdict'
    var_26 = 'from pathlib import Path as P\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'pathlib'
    var_32 = 'Path'
    var_33 = 'P'
    var_34 = 'cimport numpy\n'
    var_35 = [var_34]
    var_36 = module_0.imports(var_35)
    var_37 = list(var_36)
    var_38 = len(var_37)
    assert var_38 == 1
    var_39 = True
    var_40 = 'from typing import (\n'
    var_41 = '    List,\n'
    var_42 = '    Dict,\n'
    var_43 = ')\n'
    var_44 = [var_40, var_41, var_42, var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = 'typing'
    var_49 = 'List'
    var_50 = 3
    var_51 = 'Dict'
    var_52 = '    import os\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 1
    var_57 = True
    var_58 = '# This is a comment\n'
    var_59 = 'import sys  # inline comment\n'
    var_60 = [var_58, var_59]
    var_61 = module_0.imports(var_60)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = True
    var_65 = module_1.Config()
    var_66 = 'import numpy as numpy\n'
    var_67 = [var_66]
    var_68 = module_0.imports(var_67, var_65)
    var_69 = list(var_68)
    var_70 = len(var_69)
    assert var_70 == 1
    var_71 = 'def foo():\n'
    var_72 = '    import sys\n'
    var_73 = [var_0, var_71, var_72]
    var_74 = True
    var_75 = module_0.imports(var_73, top_only=var_74)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 1



# Parsed testcases at query #65
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
    var_17 = 'os'
    var_18 = 'path'
    var_19 = module_0.Import()
    var_20 = var_19.statement()
    assert var_20 == 'from os import path'
    var_21 = 'p'
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from os import path as p'
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



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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
    var_14 = 'defaultdict'
    var_15 = '/another/test.py'
    var_16 = str(var_10)
    assert var_16 == '/another/test.py:15 indented from collections import defaultdict'
    var_17 = 20
    var_18 = 'numpy'
    var_19 = 'np'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':20 import numpy as np'



# Parsed testcases at query #70
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



# Parsed testcases at query #71
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
    assert var_30 == '/path/to/file.py:6 import sys'



# Parsed testcases at query #72
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
    var_24 = 'array'
    var_25 = 'arr'
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from numpy import array as arr'
    var_28 = 'cf'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #73
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
    var_15 = 'import sys, os\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'cimport numpy\n'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 'from libc cimport printf\n'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = True
    var_31 = module_1.Config()
    var_32 = 'import numpy as numpy\n'
    var_33 = [var_32]
    var_34 = module_0.imports(var_33, var_31)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 1
    var_37 = 'from collections import (\n    defaultdict,\n    Counter\n)\n'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = '    import sys\n'
    var_43 = [var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 'import os  # some comment\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = 'import sys; import os\n'
    var_53 = [var_52]
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = '/some/path/file.py'
    var_58 = [var_0]
    var_59 = len(var_55)
    assert var_59 == 1
    var_60 = 'def foo():\n'
    var_61 = [var_0, var_60, var_42]
    var_62 = module_0.imports(var_61, top_only=var_30)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1



# Parsed testcases at query #74
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
    var_9 = 'os.path'
    var_10 = 'join'
    var_11 = module_0.Import()
    var_12 = var_11.statement()
    assert var_12 == 'from os.path import join'
    var_13 = 'path_join'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'from os.path import join as path_join'
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
    var_24 = 'func'
    var_25 = True
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from cython cimport func'
    var_28 = 'cf'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport func as cf'



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'numpy'
    var_4 = 'np'
    var_5 = 'cython'
    var_6 = True
    var_7 = 'cy'
    var_8 = True
    var_9 = 'path'
    var_10 = 'collections'
    var_11 = 'defaultdict'
    var_12 = 'dd'
    var_13 = 'cfunc'
    var_14 = True
    var_15 = 'cf'
    var_16 = True



# Parsed testcases at query #76
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
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 import numpy as np'
    var_15 = 4
    var_16 = 'cython'
    var_17 = True
    var_18 = module_0.Import()
    var_19 = str(var_18)
    assert var_19 == ':4 cimport cython'
    var_20 = 5
    var_21 = True
    var_22 = 'cy'
    var_23 = True
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 indented cimport cython as cy'
    var_26 = 6
    var_27 = 'path'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 from os import path'
    var_30 = 7
    var_31 = True
    var_32 = 'collections'
    var_33 = 'defaultdict'
    var_34 = 'dd'
    var_35 = module_0.Import()
    var_36 = str(var_35)
    assert var_36 == ':7 indented from collections import defaultdict as dd'
    var_37 = 8
    var_38 = '/path/to/file.py'
    var_39 = str(var_35)
    assert var_39 == '/path/to/file.py:8 import sys'



# Parsed testcases at query #77
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
    var_30 = 'from typing import (\n'
    var_31 = '    List,\n'
    var_32 = '    Dict,\n'
    var_33 = ')\n'
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = '    import os\n'
    var_39 = [var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 1
    var_43 = 'import os  # some comment\n'
    var_44 = [var_43]
    var_45 = module_0.imports(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 1
    var_48 = '/some/path/file.py'
    var_49 = [var_0]
    var_50 = len(var_46)
    assert var_50 == 1
    var_51 = True
    var_52 = module_1.Config()
    var_53 = 'import os as os\n'
    var_54 = [var_53]
    var_55 = module_0.imports(var_54, var_52)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = 'def foo():\n'
    var_59 = '    import sys\n'
    var_60 = [var_0, var_58, var_59]
    var_61 = module_0.imports(var_60, top_only=var_51)
    var_62 = list(var_61)
    var_63 = len(var_62)
    assert var_63 == 1
    var_64 = '# comment\n'
    var_65 = [var_64, var_0]
    var_66 = module_0.imports(var_65)
    var_67 = list(var_66)
    var_68 = len(var_67)
    assert var_68 == 1
    var_69 = 'yield\n'
    var_70 = [var_69, var_0]
    var_71 = module_0.imports(var_70)
    var_72 = list(var_71)
    var_73 = len(var_72)
    assert var_73 == 1
    var_74 = 'raise Exception\n'
    var_75 = [var_74, var_0]
    var_76 = module_0.imports(var_75)
    var_77 = list(var_76)
    var_78 = len(var_77)
    assert var_78 == 1
    var_79 = 'import os; import sys\n'
    var_80 = [var_79]
    var_81 = module_0.imports(var_80)
    var_82 = list(var_81)
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = 'import os \\\n'
    var_85 = '    , sys\n'
    var_86 = [var_84, var_85]
    var_87 = module_0.imports(var_86)
    var_88 = list(var_87)
    var_89 = len(var_88)
    assert var_89 == 2



# Parsed testcases at query #78
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
    var_20 = 'view'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'from cython cimport view'
    var_24 = 'array'
    var_25 = 'arr'
    var_26 = module_0.Import()
    var_27 = var_26.statement()
    assert var_27 == 'from numpy import array as arr'
    var_28 = 'v'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from cython cimport view as v'



# Parsed testcases at query #79
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import numpy as np\n'
    var_2 = 'from collections import defaultdict\n'
    var_3 = 'from pathlib import Path as P\n'
    var_4 = 'import sys, os\n'
    var_5 = 'cimport numpy\n'
    var_6 = 'from libc cimport printf\n'
    var_7 = 'from collections import (\n    defaultdict,\n    OrderedDict,\n)\n'
    var_8 = '    import sys\n'
    var_9 = 'import os  # Operating system interfaces\n'
    var_10 = 'import sys; import os\n'
    var_11 = True
    var_12 = module_0.Config()
    var_13 = 'import numpy as numpy\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = 'import os\n'
    var_16 = '/test/file.py'
    var_17 = False
    var_18 = 'os'
    var_19 = 'operating_system'



# Parsed testcases at query #80
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
    var_23 = 'cfunc'
    var_24 = True
    var_25 = module_0.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from cython cimport cfunc'
    var_27 = 'cf'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = var_29.statement()
    assert var_30 == 'from cython cimport cfunc as cf'



# Parsed testcases at query #81
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'numpy'
    var_3 = 'array'
    var_4 = 'np'
    var_5 = True
    var_6 = 'test.py'
    var_7 = 2
    var_8 = True
    var_9 = 'os'
    var_10 = None
    var_11 = module_0.Import()
    var_12 = str(var_11)
    assert var_12 == ':2 indented import os'
    var_13 = 3
    var_14 = 'sys'
    var_15 = 'path'
    var_16 = 'example.py'
    var_17 = str(var_11)
    assert var_17 == 'example.py:3 import from sys path'
    var_18 = 4
    var_19 = True
    var_20 = 'pandas'
    var_21 = 'DataFrame'
    var_22 = 'pd'
    var_23 = True
    var_24 = 'script.py'
    var_25 = str(var_11)
    assert var_25 == 'script.py:4 indented cimport from pandas DataFrame as pd'



# Parsed testcases at query #82
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == '1: import os'
    var_5 = 2
    var_6 = 'numpy'
    var_7 = 'np'
    var_8 = module_0.Import()
    var_9 = str(var_8)
    assert var_9 == '2: import numpy as np'
    var_10 = 3
    var_11 = 'sys'
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == '3: from sys import path'
    var_15 = 4
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == '4: from collections import defaultdict as dd'
    var_21 = 5
    var_22 = True
    var_23 = 'typing'
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == '5: indented import typing'
    var_26 = 6
    var_27 = 'cython'
    var_28 = True
    var_29 = module_0.Import()
    var_30 = str(var_29)
    assert var_30 == '6: cimport cython'
    var_31 = 7
    var_32 = 'pathlib'
    var_33 = '/tmp/test.py'
    var_34 = str(var_29)
    assert var_34 == '/tmp/test.py:7: import pathlib'
    var_35 = 8
    var_36 = True
    var_37 = 'asyncio'
    var_38 = 'coroutine'
    var_39 = 'aco'
    var_40 = True
    var_41 = str(var_29)
    assert var_41 == '/tmp/test.py:8: indented from asyncio cimport coroutine as aco'



# Parsed testcases at query #83
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 'os'
    var_3 = 'path'
    var_4 = 'osp'
    var_5 = '/path/to/file.py'
    var_6 = 5
    var_7 = True
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 indented import sys'
    var_12 = 15
    var_13 = 'numpy'
    var_14 = 'array'
    var_15 = 'np_array'
    var_16 = '/path/to/file.pyx'
    var_17 = str(var_10)
    assert var_17 == '/path/to/file.pyx:15 from numpy cimport array as np_array'
    var_18 = 20
    var_19 = 'math'
    var_20 = str(var_10)
    assert var_20 == '/path/to/file.py:20 import math'
    var_21 = 25
    var_22 = 'ctypes'
    var_23 = str(var_10)
    assert var_23 == '/path/to/file.pyx:25 indented cimport ctypes'



# Parsed testcases at query #84
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
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 import numpy as np'
    var_15 = 4
    var_16 = True
    var_17 = 'pandas'
    var_18 = 'pd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 indented import pandas as pd'
    var_21 = 5
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 from collections import defaultdict'
    var_26 = 6
    var_27 = 'dd'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 from collections import defaultdict as dd'
    var_30 = 7
    var_31 = 'cython'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = str(var_33)
    assert var_34 == ':7 cimport cython'
    var_35 = 8
    var_36 = 'func'
    var_37 = True
    var_38 = module_0.Import()
    var_39 = str(var_38)
    assert var_39 == ':8 from cython cimport func'
    var_40 = 9
    var_41 = '/path/to/file.py'
    var_42 = str(var_38)
    assert var_42 == '/path/to/file.py:9 import os'



# Parsed testcases at query #85
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
    var_30 = 'from libc cimport printf\n'
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
    var_47 = '    import sys\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 1
    var_52 = '/test/path.py'
    var_53 = [var_0]
    var_54 = len(var_50)
    assert var_54 == 1
    var_55 = 'def func():\n'
    var_56 = '    pass\n'
    var_57 = [var_0, var_55, var_56]
    var_58 = module_0.imports(var_57, top_only=var_40)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = '# comment\n'
    var_62 = '"""docstring"""\n'
    var_63 = 'import sys\n'
    var_64 = [var_61, var_0, var_62, var_63]
    var_65 = module_0.imports(var_64)
    var_66 = list(var_65)
    var_67 = len(var_66)
    assert var_67 == 2
    var_68 = 'import os; import sys\n'
    var_69 = [var_68]
    var_70 = module_0.imports(var_69)
    var_71 = list(var_70)
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = 'import os  # comment\n'
    var_74 = [var_73]
    var_75 = module_0.imports(var_74)
    var_76 = list(var_75)
    var_77 = len(var_76)
    assert var_77 == 1
    var_78 = 'from collections import defaultdict  # comment\n'
    var_79 = [var_78]
    var_80 = module_0.imports(var_79)
    var_81 = list(var_80)
    var_82 = len(var_81)
    assert var_82 == 1



# Parsed testcases at query #86
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
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == ':3 import numpy as np'
    var_15 = 4
    var_16 = True
    var_17 = 'pandas'
    var_18 = 'pd'
    var_19 = module_0.Import()
    var_20 = str(var_19)
    assert var_20 == ':4 indented import pandas as pd'
    var_21 = 5
    var_22 = 'collections'
    var_23 = 'defaultdict'
    var_24 = module_0.Import()
    var_25 = str(var_24)
    assert var_25 == ':5 from collections import defaultdict'
    var_26 = 6
    var_27 = 'dd'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':6 from collections import defaultdict as dd'
    var_30 = 7
    var_31 = 'cython'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = str(var_33)
    assert var_34 == ':7 cimport cython'
    var_35 = 8
    var_36 = 'cy'
    var_37 = True
    var_38 = module_0.Import()
    var_39 = str(var_38)
    assert var_39 == ':8 cimport cython as cy'
    var_40 = 9
    var_41 = '/path/to/file.py'
    var_42 = str(var_38)
    assert var_42 == '/path/to/file.py:9 import os'



# Parsed testcases at query #87
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = module_0.Import()
    var_4 = str(var_3)
    assert var_4 == '1: import os'
    var_5 = 5
    var_6 = True
    var_7 = 'sys'
    var_8 = 'test.py'
    var_9 = str(var_3)
    assert var_9 == 'test.py:5 indented import sys'
    var_10 = 10
    var_11 = 'numpy'
    var_12 = 'np'
    var_13 = module_0.Import()
    var_14 = str(var_13)
    assert var_14 == '10: import numpy as np'
    var_15 = 3
    var_16 = True
    var_17 = 'libc'
    var_18 = 'stdio'
    var_19 = True
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == '3: indented from libc cimport stdio'
    var_22 = 7
    var_23 = 'pandas'
    var_24 = 'DataFrame'
    var_25 = 'pd'
    var_26 = 'script.py'
    var_27 = str(var_20)
    assert var_27 == 'script.py:7 from pandas import DataFrame as pd'



# Parsed testcases at query #88
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
    var_9 = 's'
    var_10 = '/test/file.py'
    var_11 = 3
    var_12 = 'math'
    var_13 = None
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':3 indented import math'
    var_16 = 7
    var_17 = 'libc'
    var_18 = 'stdio'
    var_19 = '/test/module.pyx'
    var_20 = str(var_14)
    assert var_20 == '/test/module.pyx:7 from libc cimport stdio'
    var_21 = 'typing'
    var_22 = 'List'
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':1 from typing import List'



