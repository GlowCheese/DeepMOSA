####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of the Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = var_5.__str__()
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = 'sys'
    var_9 = 'test.py'
    var_10 = var_5.__str__()
    assert var_10 == 'test.py:5 import sys'
    var_11 = 15
    var_12 = True
    var_13 = 'json'
    var_14 = 'main.py'
    var_15 = var_5.__str__()
    assert var_15 == 'main.py:15 indented import json'
    var_16 = 20
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'app.py'
    var_20 = var_5.__str__()
    assert var_20 == 'app.py:20 from collections import defaultdict'
    var_21 = 25
    var_22 = 'numpy'
    var_23 = 'array'
    var_24 = 'arr'
    var_25 = 'script.py'
    var_26 = var_5.__str__()
    assert var_26 == 'script.py:25 from numpy import array as arr'
    var_27 = 30
    var_28 = 'pandas'
    var_29 = 'pd'
    var_30 = 'analysis.py'
    var_31 = var_5.__str__()
    assert var_31 == 'analysis.py:30 import pandas as pd'
    var_32 = 35
    var_33 = 'libc.stdlib'
    var_34 = 'cython_file.pyx'
    var_35 = var_5.__str__()
    assert var_35 == 'cython_file.pyx:35 cimport libc.stdlib'
    var_36 = 40
    var_37 = 'libc.math'
    var_38 = 'sin'
    var_39 = 'math_module.pyx'
    var_40 = var_5.__str__()
    assert var_40 == 'math_module.pyx:40 indented from libc.math cimport sin'
    var_41 = 45
    var_42 = 'libc.stdio'
    var_43 = 'printf'
    var_44 = 'print_func'
    var_45 = 'io.pyx'
    var_46 = var_5.__str__()
    assert var_46 == 'io.pyx:45 from libc.stdio cimport printf as print_func'



# Parsed testcases at query #2
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = str(var_5)
    assert var_6 == ':1 import os'
    var_7 = 5
    var_8 = 'sys'
    var_9 = '/path/to/file.py'
    var_10 = str(var_5)
    assert var_10 == '/path/to/file.py:5 import sys'
    var_11 = 10
    var_12 = True
    var_13 = 'json'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':10 indented import json'
    var_16 = 3
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'test.py'
    var_20 = str(var_14)
    assert var_20 == 'test.py:3 from collections import defaultdict'
    var_21 = 7
    var_22 = 'numpy'
    var_23 = 'array'
    var_24 = 'arr'
    var_25 = 'script.py'
    var_26 = str(var_14)
    assert var_26 == 'script.py:7 from numpy import array as arr'
    var_27 = 2
    var_28 = 'libc.math'
    var_29 = True
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':2 cimport libc.math'
    var_32 = 4
    var_33 = 'libc.stdlib'
    var_34 = 'malloc'
    var_35 = True
    var_36 = 'cython_file.pyx'
    var_37 = str(var_30)
    assert var_37 == 'cython_file.pyx:4 from libc.stdlib cimport malloc'
    var_38 = 15
    var_39 = True
    var_40 = 'typing'
    var_41 = 'Optional'
    var_42 = 'Opt'
    var_43 = 'module.py'
    var_44 = str(var_30)
    assert var_44 == 'module.py:15 indented from typing import Optional as Opt'
    var_45 = 6
    var_46 = 'pandas'
    var_47 = 'pd'
    var_48 = 'data.py'
    var_49 = str(var_30)
    assert var_49 == 'data.py:6 import pandas as pd'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys, re\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = '    import os\n'
    var_8 = 'cimport numpy\n'
    var_9 = 'from libc.stdlib cimport malloc\n'
    var_10 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_11 = 'from os import \\\n    path, \\\n    getcwd\n'
    var_12 = 'import os  # comment\n'
    var_13 = 'test.py'
    var_14 = 'import os\ndef func():\n    import sys\n'
    var_15 = True
    var_16 = '# comment\n\nimport os\n'
    var_17 = 'import os; import sys\n'
    var_18 = 'import os.path\n'



# Parsed testcases at query #4
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = 'path_join'
    var_5 = False
    var_6 = '/home/user/test.py'
    var_7 = 10
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':10 import sys'
    var_12 = 5
    var_13 = 'numpy'
    var_14 = 'array'
    var_15 = '/tmp/module.pyx'
    var_16 = str(var_10)
    assert var_16 == '/tmp/module.pyx:5 indented from numpy cimport array'
    var_17 = 'collections'
    var_18 = 'col'
    var_19 = './src/main.py'
    var_20 = str(var_10)
    assert var_20 == './src/main.py:1 import collections as col'
    var_21 = 99
    var_22 = 'libc.stdlib'
    var_23 = 'malloc'
    var_24 = 'mem_alloc'
    var_25 = '/project/ext.pyx'
    var_26 = str(var_10)
    assert var_26 == '/project/ext.pyx:99 indented from libc.stdlib cimport malloc as mem_alloc'
    var_27 = 7
    var_28 = 'json'
    var_29 = 'dumps'
    var_30 = 'test.py'
    var_31 = str(var_10)
    assert var_31 == 'test.py:7 from json import dumps'



# Parsed testcases at query #5
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the statement method of Import class.'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import os'
    var_6 = 2
    var_7 = 'numpy'
    var_8 = 'np'
    var_9 = module_0.Import()
    var_10 = var_9.statement()
    assert var_10 == 'import numpy as np'
    var_11 = 3
    var_12 = 'path'
    var_13 = module_0.Import()
    var_14 = var_13.statement()
    assert var_14 == 'from os import path'
    var_15 = 4
    var_16 = 'ospath'
    var_17 = module_0.Import()
    var_18 = var_17.statement()
    assert var_18 == 'from os import path as ospath'
    var_19 = 5
    var_20 = 'libc.stdlib'
    var_21 = True
    var_22 = module_0.Import()
    var_23 = var_22.statement()
    assert var_23 == 'cimport libc.stdlib'
    var_24 = 6
    var_25 = 'malloc'
    var_26 = True
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'from libc.stdlib cimport malloc'
    var_29 = 7
    var_30 = 'my_malloc'
    var_31 = True
    var_32 = module_0.Import()
    var_33 = var_32.statement()
    assert var_33 == 'from libc.stdlib cimport malloc as my_malloc'
    var_34 = 8
    var_35 = 'stdlib'
    var_36 = True
    var_37 = module_0.Import()
    var_38 = var_37.statement()
    assert var_38 == 'cimport libc.stdlib as stdlib'



# Parsed testcases at query #6
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = str(var_5)
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = True
    var_9 = 'sys'
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 indented import sys'
    var_12 = 15
    var_13 = 'os.path'
    var_14 = 'join'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':15 from os.path import join'
    var_17 = 20
    var_18 = 'collections'
    var_19 = 'defaultdict'
    var_20 = 'dd'
    var_21 = module_0.Import()
    var_22 = str(var_21)
    assert var_22 == ':20 from collections import defaultdict as dd'
    var_23 = 25
    var_24 = 'numpy'
    var_25 = 'np'
    var_26 = module_0.Import()
    var_27 = str(var_26)
    assert var_27 == ':25 import numpy as np'
    var_28 = 30
    var_29 = 'cython'
    var_30 = module_0.Import()
    var_31 = str(var_30)
    assert var_31 == ':30 cimport cython'
    var_32 = 35
    var_33 = 'json'
    var_34 = 'test.py'
    var_35 = str(var_30)
    assert var_35 == 'test.py:35 import json'
    var_36 = 40
    var_37 = 'typing'
    var_38 = 'List'
    var_39 = 'L'
    var_40 = '/home/user/module.py'
    var_41 = str(var_30)
    assert var_41 == '/home/user/module.py:40 indented from typing import List as L'
    var_42 = 45
    var_43 = 'libc.stdlib'
    var_44 = 'malloc'
    var_45 = 'cython_module.pyx'
    var_46 = str(var_30)
    assert var_46 == 'cython_module.pyx:45 from libc.stdlib cimport malloc'
    var_47 = 50
    var_48 = 'ndarray'
    var_49 = 'arr'
    var_50 = 'fast.pyx'
    var_51 = str(var_30)
    assert var_51 == 'fast.pyx:50 indented from numpy cimport ndarray as arr'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, sep\n'
    var_6 = '    import sys\n'
    var_7 = 'from os import (\n    path,\n    sep\n)\n'
    var_8 = 'os'
    var_9 = 'from os import path, \\\n    sep\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # operating system\n'
    var_13 = 'import os; import sys\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'x = 1\nimport os\ny = 2\n'
    var_18 = '# comment\nimport os\nimport sys\n'



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, sep'
    var_6 = 'from os import (\n    path,\n    sep\n)'
    var_7 = 'from os import \\\n    path, \\\n    sep'
    var_8 = '    import os'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os  # comment'
    var_12 = 'x = 1\nimport os\ny = 2'
    var_13 = 'import os\n\ndef func():\n    import sys'
    var_14 = True
    var_15 = 'import os'
    var_16 = 'test.py'
    var_17 = module_0.Config()
    var_18 = 'import os as os'
    var_19 = 'import os; import sys'
    var_20 = '# comment\nimport os\n\nimport sys'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, getcwd'
    var_6 = 'from os import (\n    path,\n    getcwd\n)'
    var_7 = 'import os, \\\n    sys'
    var_8 = '    import os'
    var_9 = 'cimport numpy'
    var_10 = 'import os\n\ndef foo():\n    import sys'
    var_11 = True
    var_12 = 'import os'
    var_13 = 'test.py'
    var_14 = 'import os  # comment'
    var_15 = 'import os; import sys'
    var_16 = module_0.Config()
    var_17 = 'import os as os'
    var_18 = module_0.Config()
    var_19 = 'import os as operating_system'



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, sep\n'
    var_7 = '    import json\n'
    var_8 = 'from os import (\n    path,\n    sep\n)\n'
    var_9 = 'from os import path, \\\n    sep\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # operating system\n'
    var_13 = '"""\nfrom fake import module\n"""\nimport real\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'import os; import sys\n'
    var_18 = 'import os as os\n'
    var_19 = module_0.Config()
    var_20 = module_0.Config()
    var_21 = 'import asyncio\n'



# Parsed testcases at query #11
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = var_5.__str__()
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = 'sys'
    var_9 = '/path/to/file.py'
    var_10 = var_5.__str__()
    assert var_10 == '/path/to/file.py:5 import sys'
    var_11 = 15
    var_12 = True
    var_13 = 'json'
    var_14 = module_0.Import()
    var_15 = var_14.__str__()
    assert var_15 == ':15 indented import json'
    var_16 = 20
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'test.py'
    var_20 = var_14.__str__()
    assert var_20 == 'test.py:20 from collections import defaultdict'
    var_21 = 25
    var_22 = 'typing'
    var_23 = 'Dict'
    var_24 = 'DictType'
    var_25 = module_0.Import()
    var_26 = var_25.__str__()
    assert var_26 == ':25 from typing import Dict as DictType'
    var_27 = 30
    var_28 = 'numpy'
    var_29 = module_0.Import()
    var_30 = var_29.__str__()
    assert var_30 == ':30 cimport numpy'
    var_31 = 35
    var_32 = 'libc.stdlib'
    var_33 = 'malloc'
    var_34 = 'cython_file.pyx'
    var_35 = var_29.__str__()
    assert var_35 == 'cython_file.pyx:35 from libc.stdlib cimport malloc'
    var_36 = 40
    var_37 = 'pandas'
    var_38 = 'pd'
    var_39 = 'analysis.py'
    var_40 = var_29.__str__()
    assert var_40 == 'analysis.py:40 indented import pandas as pd'
    var_41 = 45
    var_42 = 'matplotlib.pyplot'
    var_43 = 'plot'
    var_44 = 'plt_plot'
    var_45 = module_0.Import()
    var_46 = var_45.__str__()
    assert var_46 == ':45 indented from matplotlib.pyplot import plot as plt_plot'



# Parsed testcases at query #12
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = var_5.__str__()
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = True
    var_9 = 'sys'
    var_10 = module_0.Import()
    var_11 = var_10.__str__()
    assert var_11 == ':5 indented import sys'
    var_12 = 15
    var_13 = 'os.path'
    var_14 = 'join'
    var_15 = module_0.Import()
    var_16 = var_15.__str__()
    assert var_16 == ':15 from os.path import join'
    var_17 = 20
    var_18 = 'numpy'
    var_19 = 'array'
    var_20 = 'arr'
    var_21 = module_0.Import()
    var_22 = var_21.__str__()
    assert var_22 == ':20 from numpy import array as arr'
    var_23 = 25
    var_24 = 'libc.stdlib'
    var_25 = 'malloc'
    var_26 = module_0.Import()
    var_27 = var_26.__str__()
    assert var_27 == ':25 from libc.stdlib cimport malloc'
    var_28 = 30
    var_29 = 'json'
    var_30 = '/path/to/file.py'
    var_31 = var_26.__str__()
    assert var_31 == '/path/to/file.py:30 import json'
    var_32 = 8
    var_33 = 'collections'
    var_34 = 'defaultdict'
    var_35 = 'dd'
    var_36 = 'test.py'
    var_37 = var_26.__str__()
    assert var_37 == 'test.py:8 indented from collections import defaultdict as dd'
    var_38 = 12
    var_39 = 'cython'
    var_40 = 'parallel'
    var_41 = 'par'
    var_42 = 'cython_file.pyx'
    var_43 = var_26.__str__()
    assert var_43 == 'cython_file.pyx:12 indented from cython cimport parallel as par'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, environ'
    var_6 = '    import os'
    var_7 = 'from os import (\n    path,\n    environ\n)'
    var_8 = 'from os import path, \\\n    environ'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os  # comment'
    var_12 = 'test.py'
    var_13 = 'import os'
    var_14 = 'import os\n\ndef func():\n    import sys'
    var_15 = True
    var_16 = ''
    var_17 = 'x = 5\nimport os\ny = 10'
    var_18 = 'import os; import sys'
    var_19 = False
    var_20 = 'os'
    var_21 = None
    var_22 = 'path'
    var_23 = 'p'
    var_24 = 'numpy'
    var_25 = 'np'
    var_26 = 'libc.stdlib'
    var_27 = 'malloc'



# Parsed testcases at query #14
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the statement method of the Import class.'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = var_5.statement()
    assert var_6 == 'import os'
    var_7 = 2
    var_8 = 'numpy'
    var_9 = 'np'
    var_10 = module_0.Import()
    var_11 = var_10.statement()
    assert var_11 == 'import numpy as np'
    var_12 = 3
    var_13 = 'path'
    var_14 = module_0.Import()
    var_15 = var_14.statement()
    assert var_15 == 'from os import path'
    var_16 = 4
    var_17 = 'p'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from os import path as p'
    var_20 = 5
    var_21 = 'libc.stdlib'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'cimport libc.stdlib'
    var_25 = 6
    var_26 = 'malloc'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from libc.stdlib cimport malloc'
    var_30 = 7
    var_31 = 'my_malloc'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = var_33.statement()
    assert var_34 == 'from libc.stdlib cimport malloc as my_malloc'



# Parsed testcases at query #15
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the statement method of Import class.'
    var_1 = 1
    var_2 = False
    var_3 = 'os'
    var_4 = module_0.Import()
    var_5 = var_4.statement()
    assert var_5 == 'import os'
    var_6 = 2
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
    var_17 = 'dd'
    var_18 = module_0.Import()
    var_19 = var_18.statement()
    assert var_19 == 'from collections import defaultdict as dd'
    var_20 = 5
    var_21 = 'libc.stdlib'
    var_22 = True
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'cimport libc.stdlib'
    var_25 = 6
    var_26 = 'malloc'
    var_27 = True
    var_28 = module_0.Import()
    var_29 = var_28.statement()
    assert var_29 == 'from libc.stdlib cimport malloc'
    var_30 = 7
    var_31 = 'my_malloc'
    var_32 = True
    var_33 = module_0.Import()
    var_34 = var_33.statement()
    assert var_34 == 'from libc.stdlib cimport malloc as my_malloc'
    var_35 = 8
    var_36 = 'libc.math'
    var_37 = 'math'
    var_38 = True
    var_39 = module_0.Import()
    var_40 = var_39.statement()
    assert var_40 == 'cimport libc.math as math'



# Parsed testcases at query #16
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = str(var_5)
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = 'sys'
    var_9 = '/path/to/file.py'
    var_10 = str(var_5)
    assert var_10 == '/path/to/file.py:5 import sys'
    var_11 = 15
    var_12 = True
    var_13 = 'json'
    var_14 = module_0.Import()
    var_15 = str(var_14)
    assert var_15 == ':15 indented import json'
    var_16 = 20
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'test.py'
    var_20 = str(var_14)
    assert var_20 == 'test.py:20 from collections import defaultdict'
    var_21 = 25
    var_22 = 'typing'
    var_23 = 'List'
    var_24 = 'ListType'
    var_25 = module_0.Import()
    var_26 = str(var_25)
    assert var_26 == ':25 from typing import List as ListType'
    var_27 = 30
    var_28 = 'numpy'
    var_29 = 'np'
    var_30 = 'script.py'
    var_31 = str(var_25)
    assert var_31 == 'script.py:30 import numpy as np'
    var_32 = 35
    var_33 = 'cython'
    var_34 = module_0.Import()
    var_35 = str(var_34)
    assert var_35 == ':35 cimport cython'
    var_36 = 40
    var_37 = 'libc.stdlib'
    var_38 = 'malloc'
    var_39 = 'memory_alloc'
    var_40 = 'cython_file.pyx'
    var_41 = str(var_34)
    assert var_41 == 'cython_file.pyx:40 indented from libc.stdlib cimport malloc as memory_alloc'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, environ\n'
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = 'from os import \\\n    path\n'
    var_8 = '    import sys\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = 'import os  # system module\n'
    var_12 = 'import os; import sys\n'
    var_13 = 'import os\ndef foo():\n    import sys\n'
    var_14 = True
    var_15 = 'test.py'
    var_16 = '"""\nimport os\n"""\nimport sys\n'
    var_17 = module_0.Config()
    var_18 = 'import os as os\n'
    var_19 = 'import os.path\n'
    var_20 = 'from os import *\n'



# Parsed testcases at query #18
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = 'path_join'
    var_5 = False
    var_6 = '/home/user/script.py'
    var_7 = 5
    var_8 = 'sys'
    var_9 = None
    var_10 = module_0.Import()
    var_11 = str(var_10)
    assert var_11 == ':5 import sys'
    var_12 = 15
    var_13 = 'collections'
    var_14 = 'defaultdict'
    var_15 = module_0.Import()
    var_16 = str(var_15)
    assert var_16 == ':15 indented from collections import defaultdict'
    var_17 = 3
    var_18 = 'numpy'
    var_19 = 'array'
    var_20 = 'np_array'
    var_21 = '/tmp/test.pyx'
    var_22 = str(var_15)
    assert var_22 == '/tmp/test.pyx:3 from numpy cimport array as np_array'
    var_23 = 'libc.stdlib'
    var_24 = '/code/module.pyx'
    var_25 = str(var_15)
    assert var_25 == '/code/module.pyx:1 cimport libc.stdlib'
    var_26 = 20
    var_27 = 'json'
    var_28 = 'loads'
    var_29 = '/app/main.py'
    var_30 = str(var_15)
    assert var_30 == '/app/main.py:20 from json import loads'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, environ\n'
    var_6 = '    import json\n'
    var_7 = 'from typing import (\n    List,\n    Dict\n)\n'
    var_8 = 'from os import \\\n    path, environ\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = 'import os  # operating system\n'
    var_12 = 'import os\n\ndef foo():\n    import sys\n'
    var_13 = True
    var_14 = 'import os\n'
    var_15 = 'test_file.py'
    var_16 = 'import os; import sys\n'
    var_17 = '"""\nModule docstring\n"""\nimport os\n'
    var_18 = 'import os\n\nimport sys\n'
    var_19 = ''



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, environ\n'
    var_6 = '    import sys\n'
    var_7 = 'from os import (\n    path,\n    environ\n)\n'
    var_8 = 'from os import path, \\\n    environ\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = 'import os  # comment\n'
    var_12 = 'import os\n\ndef foo():\n    import sys\n'
    var_13 = True
    var_14 = 'test.py'
    var_15 = 'import os; import sys\n'
    var_16 = 'x = 1\nimport os\ny = 2\n'
    var_17 = '"""\nModule docstring\nimport fake\n"""\nimport os\n'



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path'
    var_3 = 'import numpy as np'
    var_4 = 'from collections import OrderedDict as OD'
    var_5 = 'from os import path, environ'
    var_6 = '    import json'
    var_7 = 'from os import (\n    path,\n    environ\n)'
    var_8 = 'from os import path, \\\n    environ'
    var_9 = 'cimport cython'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'test.py'
    var_12 = 'import os'
    var_13 = 'import os\ndef foo():\n    import sys'
    var_14 = True
    var_15 = 'import os  # comment'
    var_16 = 'x = 5\nimport os\ny = 10'
    var_17 = 'import os; import sys'
    var_18 = module_0.Config()
    var_19 = 'import os as os'
    var_20 = False
    var_21 = module_0.Config()



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'from os import path\nfrom sys import argv\n'
    var_3 = 'import numpy as np\nfrom os import path as p\n'
    var_4 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_5 = 'from os import path, \\\n    getcwd\n'
    var_6 = '    import os\n'
    var_7 = 'cimport numpy\nfrom libc.stdlib cimport malloc\n'
    var_8 = 'import os, sys, json\n'
    var_9 = 'import os  # comment\nfrom sys import argv  # another comment\n'
    var_10 = 'import os\n\ndef foo():\n    import sys\n'
    var_11 = True
    var_12 = 'import os\n'
    var_13 = 'test.py'
    var_14 = module_0.Config()
    var_15 = 'import os as os\nfrom sys import argv as argv\n'
    var_16 = 'from os import path, getcwd, listdir\n'
    var_17 = 'x = 1\nimport os\ny = 2\nimport sys\n'
    var_18 = ''



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = '    import os\n'
    var_8 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_9 = 'from os import path, \\\n    getcwd\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # comment\n'
    var_13 = 'import os; import sys\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'import os\nclass MyClass:\n    import sys\n'
    var_17 = 'test_file.py'
    var_18 = 'import os\n\nimport sys\n'
    var_19 = module_0.Config()
    var_20 = 'import os as os\n'
    var_21 = module_0.Config()
    var_22 = 'from os import path as path\n'
    var_23 = 'yield\n    x\nimport os\n'
    var_24 = ''



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os.path import join as path_join\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, environ\n'
    var_7 = '    import json\n'
    var_8 = 'from module import (\n    func1,\n    func2\n)\n'
    var_9 = 'from module import \\\n    func1, func2\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'test.py'
    var_13 = 'import os\n\ndef func():\n    import sys\n'
    var_14 = True
    var_15 = 'import os  # this is a comment\n'
    var_16 = 'import os; import sys\n'
    var_17 = '# comment\nimport os\nimport sys\n'
    var_18 = 'yield\nimport os\n'
    var_19 = module_0.Config()
    var_20 = 'import os as os\n'
    var_21 = 'from os import path as path\n'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'import os, sys, json'
    var_6 = 'from os import (\n    path,\n    getcwd\n)'
    var_7 = 'from os import path, \\\n    getcwd'
    var_8 = '    import os'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os'
    var_12 = 'test.py'
    var_13 = 'import os\n\ndef foo():\n    import sys'
    var_14 = True
    var_15 = 'import os  # comment'
    var_16 = 'import os; import sys'
    var_17 = '"""\nimport fake\n"""\nimport os'
    var_18 = module_0.Config()
    var_19 = 'import os as os'
    var_20 = module_0.Config()
    var_21 = 'import os as operating_system'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 'Test the imports function with various import scenarios.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, environ\n'
    var_7 = '    import json\n'
    var_8 = 'from os import (\n    path,\n    environ\n)\n'
    var_9 = 'from os import path, \\\n    environ\n'
    var_10 = 'import os  # operating system\n'
    var_11 = 'cimport numpy\n'
    var_12 = 'from libc.stdlib cimport malloc\n'
    var_13 = 'import sys\n'
    var_14 = '/test/file.py'
    var_15 = 'import os\ndef foo():\n    import sys\n'
    var_16 = True
    var_17 = 'x = 5\nimport os\ny = 10\n'
    var_18 = '\nimport os\n'
    var_19 = module_0.Config()
    var_20 = 'import os as os\n'
    var_21 = 'import os as o, sys\n'
    var_22 = False
    var_23 = 'os'
    var_24 = None
    var_25 = module_1.Import()
    var_26 = var_25.statement()
    assert var_26 == 'import os'
    var_27 = 'collections'
    var_28 = 'OrderedDict'
    var_29 = 'OD'
    var_30 = module_1.Import()
    var_31 = var_30.statement()
    assert var_31 == 'from collections import OrderedDict as OD'
    var_32 = 5
    var_33 = 'test.py'
    var_34 = str(var_30)
    var_35 = str(var_30)
    var_36 = str(var_30)



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, sep'
    var_6 = '    import os'
    var_7 = 'from os import (\n    path,\n    sep\n)'
    var_8 = 'from os import \\\n    path, \\\n    sep'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os  # comment\nimport sys'
    var_12 = '"""\nimport os\n"""\nimport sys'
    var_13 = 'import os\n\ndef foo():\n    import sys'
    var_14 = True
    var_15 = 'import os'
    var_16 = 'test.py'
    var_17 = ''
    var_18 = module_0.Config()
    var_19 = 'import os as os'
    var_20 = 'from os import path as p, sep as s'
    var_21 = False
    var_22 = 'os'
    var_23 = 'path'
    var_24 = 'p'
    var_25 = module_1.Import()
    var_26 = var_25.statement()
    assert var_26 == 'from os import path as p'
    var_27 = str(var_25)
    var_28 = str(var_25)



# Parsed testcases at query #3
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of the Import class.'
    var_1 = 10
    var_2 = True
    var_3 = 'os.path'
    var_4 = 'join'
    var_5 = 'path_join'
    var_6 = False
    var_7 = '/home/user/test.py'
    var_8 = 5
    var_9 = 'sys'
    var_10 = None
    var_11 = '/home/user/main.py'
    var_12 = 3
    var_13 = 'numpy'
    var_14 = 'array'
    var_15 = '/home/user/cython_module.pyx'
    var_16 = 15
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'dd'
    var_20 = module_0.Import()
    var_21 = str(var_20)
    assert var_21 == ':15 indented from collections import defaultdict as dd'
    var_22 = 20
    var_23 = 'mymodule'
    var_24 = '/test/file.pyx'
    var_25 = str(var_20)
    assert var_25 == '/test/file.pyx:20 indented cimport mymodule'
    var_26 = 'np'
    var_27 = '/project/main.py'
    var_28 = str(var_20)
    assert var_28 == '/project/main.py:1 import numpy as np'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'from os import path, sep\n'
    var_6 = 'from os import (\n    path,\n    sep\n)\n'
    var_7 = 'from os import \\\n    path, \\\n    sep\n'
    var_8 = '    import os\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = 'import os\n'
    var_12 = 'test.py'
    var_13 = 'import os\n\ndef foo():\n    import sys\n'
    var_14 = True
    var_15 = 'import os  # operating system\n'
    var_16 = 'import os; import sys\n'
    var_17 = module_0.Config()
    var_18 = 'from os import path as path\n'
    var_19 = False
    var_20 = module_0.Config()
    var_21 = 'import os.path\n'
    var_22 = '\n# comment\nimport os\n'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = 'operating_system'
    var_4 = 'path'
    var_5 = 'p'
    var_6 = 'numpy'
    var_7 = True
    var_8 = 'np'
    var_9 = True
    var_10 = 'libc.stdlib'
    var_11 = 'malloc'
    var_12 = True
    var_13 = 'mem_alloc'
    var_14 = True
    var_15 = 5
    var_16 = True
    var_17 = 'sys'
    var_18 = 'package.subpackage.module'
    var_19 = 'MyClass'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path'
    var_3 = 'import numpy as np'
    var_4 = 'from os.path import join as path_join'
    var_5 = 'from collections import Counter, defaultdict'
    var_6 = 'from typing import (\n    Dict,\n    List\n)'
    var_7 = 'from os import \\\n    path, \\\n    environ'
    var_8 = '    import json'
    var_9 = 'import re  # regular expressions'
    var_10 = 'cimport numpy'
    var_11 = 'from libc.stdlib cimport malloc'
    var_12 = '# comment\nimport os\n\nimport sys'
    var_13 = 'import os\n\ndef foo():\n    import sys'
    var_14 = True
    var_15 = 'import os'
    var_16 = 'test.py'
    var_17 = 0



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, environ\n'
    var_6 = 'from os import (\n    path,\n    environ\n)\n'
    var_7 = '    import os\n'
    var_8 = 'import os  # comment\n'
    var_9 = 'cimport cython\n'
    var_10 = 'from libc.stdio cimport printf\n'
    var_11 = 'import os\n\ndef foo():\n    import sys\n'
    var_12 = True
    var_13 = 'test.py'
    var_14 = 'import os\n'
    var_15 = 'x = 1\nimport os\ny = 2\n'
    var_16 = 'import os, \\\n    sys\n'
    var_17 = 'import os; import sys\n'
    var_18 = module_0.Config()
    var_19 = 'import os as os\n'
    var_20 = 'from os.path import join\n'



# Parsed testcases at query #8
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = module_0.Import()
    var_6 = var_5.__str__()
    assert var_6 == ':10 import os'
    var_7 = 5
    var_8 = 'sys'
    var_9 = 'test.py'
    var_10 = var_5.__str__()
    assert var_10 == 'test.py:5 import sys'
    var_11 = 15
    var_12 = True
    var_13 = 'json'
    var_14 = module_0.Import()
    var_15 = var_14.__str__()
    assert var_15 == ':15 indented import json'
    var_16 = 20
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = 'app.py'
    var_20 = var_14.__str__()
    assert var_20 == 'app.py:20 from collections import defaultdict'
    var_21 = 25
    var_22 = 'numpy'
    var_23 = 'array'
    var_24 = 'arr'
    var_25 = 'script.py'
    var_26 = var_14.__str__()
    assert var_26 == 'script.py:25 from numpy import array as arr'
    var_27 = 30
    var_28 = 'libc.stdlib'
    var_29 = 'cython_module.pyx'
    var_30 = var_14.__str__()
    assert var_30 == 'cython_module.pyx:30 cimport libc.stdlib'
    var_31 = 35
    var_32 = 'libc.math'
    var_33 = 'sin'
    var_34 = 'sine'
    var_35 = 'math.pyx'
    var_36 = var_14.__str__()
    assert var_36 == 'math.pyx:35 indented from libc.math cimport sin as sine'
    var_37 = 40
    var_38 = 'pandas'
    var_39 = 'pd'
    var_40 = module_0.Import()
    var_41 = var_40.__str__()
    assert var_41 == ':40 import pandas as pd'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os.path import join as path_join\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from pathlib import Path, PurePath\n'
    var_7 = '    import os\n'
    var_8 = 'from pathlib import (\n    Path,\n    PurePath\n)\n'
    var_9 = 'from pathlib import \\\n    Path, \\\n    PurePath\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # system module\n'
    var_13 = "x = 1\nimport os\nprint('hello')\n"
    var_14 = 'import os\n\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = False
    var_18 = 'os'
    var_19 = 'pathlib'
    var_20 = 'Path'
    var_21 = 'numpy'
    var_22 = 'np'
    var_23 = 'libc.stdlib'
    var_24 = 'malloc'
    var_25 = 5



# Parsed testcases at query #10
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the __str__ method of the Import class.'
    var_1 = 10
    var_2 = False
    var_3 = 'os'
    var_4 = None
    var_5 = '/path/to/file.py'
    var_6 = 5
    var_7 = True
    var_8 = 'sys'
    var_9 = 15
    var_10 = 'os.path'
    var_11 = 'join'
    var_12 = 20
    var_13 = 'numpy'
    var_14 = 'np'
    var_15 = 25
    var_16 = 'collections'
    var_17 = 'defaultdict'
    var_18 = 'dd'
    var_19 = 30
    var_20 = 'ndarray'
    var_21 = 35
    var_22 = 'json'
    var_23 = module_0.Import()
    var_24 = str(var_23)
    assert var_24 == ':35 import json'
    var_25 = 40
    var_26 = 're'
    var_27 = 'regex'
    var_28 = module_0.Import()
    var_29 = str(var_28)
    assert var_29 == ':40 indented import re as regex'
    var_30 = 45
    var_31 = 'libc.stdlib'
    var_32 = 'malloc'
    var_33 = 'mem_alloc'
    var_34 = 'cython_file.pyx'
    var_35 = str(var_28)
    assert var_35 == 'cython_file.pyx:45 indented from libc.stdlib cimport malloc as mem_alloc'



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, sep\n'
    var_7 = '    import os\n'
    var_8 = 'from os import (\n    path,\n    sep\n)\n'
    var_9 = 'from os import \\\n    path, \\\n    sep\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc cimport math\n'
    var_12 = 'import os  # comment\n'
    var_13 = '"""\nModule docstring\n"""\nimport os\n'
    var_14 = 'import os\n\ndef func():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'import os; import sys\n'
    var_18 = ''
    var_19 = module_0.Config()
    var_20 = 'import os as os\n'
    var_21 = module_0.Config()
    var_22 = 'from os import path as path\n'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, sep\n'
    var_7 = 'from os import (\n    path,\n    sep\n)\n'
    var_8 = 'from os import \\\n    path, \\\n    sep\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = '    import os\n'
    var_12 = 'import os  # comment\n'
    var_13 = '# comment\nimport os\n\nimport sys\n'
    var_14 = 'import os\n\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'import os\n\nclass Foo:\n    import sys\n'
    var_17 = 'test.py'
    var_18 = module_0.Config()
    var_19 = 'import os as os\n'
    var_20 = module_0.Config()
    var_21 = 'from os import path as path\n'
    var_22 = 'import os; import sys\n'
    var_23 = 'import os.path\n'
    var_24 = 'import os\nimport sys\nimport json\n'



# Parsed testcases at query #13
#--------------------------


import isort.identify as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'from os import path, environ\n'
    var_6 = '    import sys\n'
    var_7 = 'from os import (\n    path,\n    environ\n)\n'
    var_8 = 'from os import path, \\\n    environ\n'
    var_9 = 'import os  # operating system\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os\n\ndef foo():\n    import sys\n'
    var_13 = True
    var_14 = 'test.py'
    var_15 = 'import os; import sys\n'
    var_16 = '"""\nimport fake\n"""\nimport real\n'
    var_17 = False
    var_18 = 'os'
    var_19 = None
    var_20 = module_0.Import()
    var_21 = var_20.statement()
    assert var_21 == 'import os'
    var_22 = 'path'
    var_23 = 'p'
    var_24 = module_0.Import()
    var_25 = var_24.statement()
    assert var_25 == 'from os import path as p'
    var_26 = 5
    var_27 = 'sys'
    var_28 = str(var_24)
    var_29 = str(var_24)
    var_30 = str(var_24)



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path\nfrom typing import List'
    var_3 = 'import numpy as np\nfrom collections import defaultdict as dd'
    var_4 = 'from os import (\n    path,\n    environ\n)'
    var_5 = 'from sys import \\\n    argv, \\\n    exit'
    var_6 = 'if True:\n    import os'
    var_7 = 'import os; import sys'
    var_8 = 'import os  # important module'
    var_9 = 'from os import path, environ, getcwd'
    var_10 = 'os'
    var_11 = 'import os\n\ndef func():\n    import sys'
    var_12 = True
    var_13 = 'cimport numpy\nfrom libc.stdlib cimport malloc'
    var_14 = 'test.py'
    var_15 = 'import os'
    var_16 = 'yield\nimport os'
    var_17 = ''
    var_18 = module_0.Config()
    var_19 = 'import os as os\nfrom sys import argv as argv'
    var_20 = module_0.Config()
    var_21 = 'import os as operating_system'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys, json\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = '    import os\n'
    var_8 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_9 = 'from os import path, \\\n    getcwd\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # comment\n'
    var_13 = 'x = 5\nimport os\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'import os; import sys\n'
    var_18 = module_0.Config()
    var_19 = 'import os as os\n'
    var_20 = module_0.Config()
    var_21 = 'from os import path as path\n'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import os as operating_system\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'from os import path, environ\n'
    var_6 = '    import sys\n'
    var_7 = 'from os import (\n    path,\n    environ\n)\n'
    var_8 = 'from os import \\\n    path, \\\n    environ\n'
    var_9 = 'import os; import sys\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc cimport stdlib\n'
    var_12 = 'import os  # this is a comment\n'
    var_13 = 'import os\n\nimport sys\n'
    var_14 = 'import os\n\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = ''
    var_18 = 'x = 5\nimport os\nprint(x)\n'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path'
    var_3 = 'import numpy as np'
    var_4 = 'from collections import OrderedDict as OD'
    var_5 = 'from os import path, environ'
    var_6 = 'from os import (\n    path,\n    environ\n)'
    var_7 = 'from os import \\\n    path'
    var_8 = '    import os'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os  # this is a comment'
    var_12 = 'import os\n\ndef foo():\n    import sys'
    var_13 = True
    var_14 = 'test.py'
    var_15 = 'import os'
    var_16 = 'import os; import sys'
    var_17 = module_0.Config()
    var_18 = 'import os as os'
    var_19 = module_0.Config()
    var_20 = 'import os as operating_system'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from pathlib import Path'
    var_3 = 'import numpy as np'
    var_4 = 'from collections import OrderedDict as OD'
    var_5 = 'from os import path, environ, getcwd'
    var_6 = 'os'
    var_7 = '    import json'
    var_8 = 'from collections import (\n    namedtuple,\n    defaultdict\n)'
    var_9 = 'from os import \\\n    path, \\\n    environ'
    var_10 = 'import os  # for operating system'
    var_11 = 'cimport numpy'
    var_12 = 'from libc.stdlib cimport malloc, free'
    var_13 = 'import os'
    var_14 = '/test/file.py'
    var_15 = 'import os\ndef foo():\n    import sys'
    var_16 = True
    var_17 = 'x = 5\nimport os\ny = 10'
    var_18 = 'import os; import sys'
    var_19 = module_0.Config()
    var_20 = 'import os as os'



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function for parsing various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import OrderedDict as OD\n'
    var_5 = 'import os, sys, json\n'
    var_6 = 'from os import path, getcwd, environ\n'
    var_7 = 'os'
    var_8 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_9 = 'import os, \\\n    sys\n'
    var_10 = '    import json\n'
    var_11 = 'cimport numpy\n'
    var_12 = 'from libc.stdlib cimport malloc\n'
    var_13 = 'import os  # comment\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'x = 1\nimport os\ny = 2\n'
    var_18 = 'import os; import sys\n'
    var_19 = 'import os as os\n'
    var_20 = module_0.Config()



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, environ'
    var_6 = '    import os'
    var_7 = 'from os import (\n    path,\n    environ\n)'
    var_8 = 'from os import \\\n    path, \\\n    environ'
    var_9 = 'cimport numpy'
    var_10 = 'from libc cimport stdlib'
    var_11 = 'import os  # this is a comment\nimport sys'
    var_12 = 'import os\ndef foo():\n    import sys'
    var_13 = True
    var_14 = 'test.py'
    var_15 = 'import os'
    var_16 = '"""\nimport os\n"""\nimport sys'
    var_17 = 'import os; import sys'
    var_18 = False
    var_19 = 'os'
    var_20 = 'path'
    var_21 = 'p'
    var_22 = 'numpy'
    var_23 = 'np'



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = '    import os\n'
    var_8 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_9 = 'from os import path,\\\n    getcwd\n'
    var_10 = 'cimport numpy\n'
    var_11 = 'from libc.stdlib cimport malloc\n'
    var_12 = 'import os  # comment\n'
    var_13 = '"""\nimport fake\n"""\nimport os\n'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'test.py'
    var_17 = 'import os\n\nimport sys\n'
    var_18 = module_0.Config()
    var_19 = 'import os as os\n'
    var_20 = module_0.Config()
    var_21 = 'from os import path as path\n'



# Parsed testcases at query #22
#--------------------------


import isort.identify as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\nimport sys'
    var_2 = 'from os import path'
    var_3 = 'import numpy as np'
    var_4 = 'from os import path as p'
    var_5 = 'from os import path, getcwd'
    var_6 = '    import os'
    var_7 = 'from os import (\n    path,\n    getcwd\n)'
    var_8 = 'from os import \\\n    path, \\\n    getcwd'
    var_9 = 'cimport numpy'
    var_10 = 'from libc.stdlib cimport malloc'
    var_11 = 'import os  # comment'
    var_12 = 'import os\n\ndef foo():\n    import sys'
    var_13 = True
    var_14 = 'test.py'
    var_15 = 'import os'
    var_16 = 'yield\nimport os'
    var_17 = 'x = 1; import os'
    var_18 = False
    var_19 = 'os'
    var_20 = 'path'
    var_21 = 'p'
    var_22 = None
    var_23 = module_0.Import()
    var_24 = var_23.statement()
    assert var_24 == 'from os import path as p'
    var_25 = 'numpy'
    var_26 = 'np'
    var_27 = module_0.Import()
    var_28 = var_27.statement()
    assert var_28 == 'import numpy as np'
    var_29 = 5
    var_30 = str(var_27)
    var_31 = str(var_27)
    var_32 = ''
    var_33 = module_1.Config()
    var_34 = 'from os import path as path'
    var_35 = module_1.Config()



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from collections import defaultdict as dd\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_7 = '    import json\n'
    var_8 = 'cimport numpy\n'
    var_9 = 'from libc.stdlib cimport malloc\n'
    var_10 = 'import os  # operating system\n'
    var_11 = 'from os import \\\n    path\n'
    var_12 = 'import os; import sys\n'
    var_13 = 'import os\ndef foo():\n    import sys\n'
    var_14 = True
    var_15 = 'test.py'
    var_16 = 'from os import path, getcwd, environ\n'
    var_17 = "x = 1\nprint('hello')\nimport os\n"
    var_18 = 'import os\n\nimport sys\n'



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from pathlib import Path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os.path import join as path_join\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_8 = 'from os import path, \\\n    getcwd\n'
    var_9 = 'cimport numpy\n'
    var_10 = 'from libc.stdlib cimport malloc\n'
    var_11 = '    import os\n'
    var_12 = 'import sys\n'
    var_13 = 'test.py'
    var_14 = 'import os\n\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'import os\n\nclass Foo:\n    import sys\n'
    var_17 = '# import os\n'
    var_18 = 'import os  # operating system\n'
    var_19 = ''
    var_20 = 'import os; import sys\n'
    var_21 = module_0.Config()
    var_22 = 'import os as os\n'
    var_23 = module_0.Config()
    var_24 = 'from os import path as path\n'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the imports function with various import statements.'
    var_1 = 'import os\n'
    var_2 = 'from os import path\n'
    var_3 = 'import numpy as np\n'
    var_4 = 'from os import path as p\n'
    var_5 = 'import os, sys\n'
    var_6 = 'from os import path, getcwd\n'
    var_7 = '    import os\n'
    var_8 = 'cimport numpy\n'
    var_9 = 'from libc.stdlib cimport malloc\n'
    var_10 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_11 = 'from os import path, \\\n    getcwd\n'
    var_12 = '# comment\nimport os\n'
    var_13 = 'test.py'
    var_14 = 'import os\ndef foo():\n    import sys\n'
    var_15 = True
    var_16 = 'import os\nclass Foo:\n    import sys\n'
    var_17 = 'import os  # comment\n'
    var_18 = 'import os\nraise ImportError\nimport sys\n'
    var_19 = 'import os; import sys\n'



