####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_4 = 'from x import y'
    var_5 = module_0.strip_syntax(var_4)
    assert var_5 == 'x y'
    var_6 = 'cimport numpy'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'numpy'
    var_8 = 'from numpy cimport array'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'numpy array'
    var_10 = 'import os, \\\n    sys'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == 'os sys'
    var_12 = 'from module import (func1, func2)'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == 'module func1 func2'
    var_14 = 'import (os, sys)'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == 'os sys'
    var_16 = 'from module import { func1, func2 }'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == 'module {|func1 func2|}'
    var_18 = 'import _import'
    var_19 = module_0.strip_syntax(var_18)
    assert var_19 == '[[i]]'
    var_20 = 'import _cimport'
    var_21 = module_0.strip_syntax(var_20)
    assert var_21 == '[[ci]]'
    var_22 = 'from module import _import, _cimport'
    var_23 = module_0.strip_syntax(var_22)
    assert var_23 == 'module [[i]] [[ci]]'
    var_24 = 'from module import (func1, func2), \\\n    func3'
    var_25 = module_0.strip_syntax(var_24)
    assert var_25 == 'module func1 func2 func3'
    var_26 = 'import'
    var_27 = module_0.strip_syntax(var_26)
    assert var_27 == ''
    var_28 = 'from'
    var_29 = module_0.strip_syntax(var_28)
    assert var_29 == ''
    var_30 = ''
    var_31 = module_0.strip_syntax(var_30)
    assert var_31 == ''
    var_32 = 'import  os,  sys'
    var_33 = module_0.strip_syntax(var_32)
    assert var_33 == 'os sys'
    var_34 = 'import _import_test'
    var_35 = module_0.strip_syntax(var_34)



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import collections as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\\\n    func1,\\\n    func2)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import func1, func2,\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort:imports-STDLIB\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = module_1.OrderedDict()
    var_17 = 'import os\r\nimport sys\r\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = True
    var_20 = module_2.Config()
    var_21 = 'import os\n'
    var_22 = module_0.file_contents(var_21, var_20)
    var_23 = var_22.verbose_output
    var_24 = len(var_23)
    var_25 = '# above comment\nimport os\n'
    var_26 = module_0.file_contents(var_25)
    var_27 = 'from module import (\\\n    func1,  # nested comment\\\n    func2)\n'
    var_28 = module_0.file_contents(var_27)



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import collections as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # system module\nimport sys  # system module\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from very.long.package.name import (\n    module1,\n    module2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from numpy import array as arr\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from collections import (\n    defaultdict,\n    OrderedDict,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = 'test_module'
    var_17 = [var_16]
    var_18 = module_1.Config()
    var_19 = 'import test_module\nimport os\n'
    var_20 = module_0.file_contents(var_19, var_18)
    var_21 = ''
    var_22 = module_0.file_contents(var_21)
    var_23 = module_2.OrderedDict()
    var_24 = '# This is a comment\n# Another comment\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = 'import os\r\nimport sys\r\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = True
    var_31 = module_1.Config()
    var_32 = 'import os\n'
    var_33 = module_0.file_contents(var_32, var_31)
    var_34 = var_33.verbose_output
    var_35 = len(var_34)
    var_36 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = '# Above comment\nimport os\n'
    var_39 = module_0.file_contents(var_38)
    var_40 = 'import os  # isort:skip\nimport sys\n'
    var_41 = module_0.file_contents(var_40)
    var_42 = module_1.Config()
    var_43 = 'from module import func as f  # comment\n'
    var_44 = module_0.file_contents(var_43, var_42)
    var_45 = module_1.Config()
    var_46 = 'import os as os\n'
    var_47 = module_0.file_contents(var_46, var_45)
    var_48 = module_1.Config()
    var_49 = "print('hello')\nimport os\n"
    var_50 = module_0.file_contents(var_49, var_48)
    var_51 = 'FIRST'
    var_52 = 'SECOND'
    var_53 = [var_51, var_52]
    var_54 = module_1.Config()
    var_55 = 'import unknown_module\n'
    var_56 = module_0.file_contents(var_55, var_54)
    var_57 = 'import os; import sys\n'
    var_58 = module_0.file_contents(var_57)
    var_59 = 'from module import \\\n    func1, \\\n    func2\n'
    var_60 = module_0.file_contents(var_59)



# Parsed testcases at query #4
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\\\n    function1,\\\n    function2)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import function1, function2,\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'tests'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'import pytest\nimport os\n'
    var_16 = module_0.file_contents(var_15, var_14)
    var_17 = '# isort:imports-firstparty\nimport mymodule\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = "print('hello')\nimport os\n"
    var_20 = module_0.file_contents(var_19)
    var_21 = ''
    var_22 = module_0.file_contents(var_21)
    var_23 = var_22.imports
    var_24 = len(var_23)
    var_25 = '# Just a comment\n# Another comment\n'
    var_26 = module_0.file_contents(var_25)
    var_27 = var_26.lines_without_imports
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'from module import (\\\n    func1,  # comment1\\\n    func2  # comment2\\\n)\n'
    var_30 = module_0.file_contents(var_29)
    var_31 = '# STDLIB'
    var_32 = '# THIRDPARTY'
    var_33 = [var_31, var_32]
    var_34 = module_1.Config()
    var_35 = '# STDLIB\nimport os\n# THIRDPARTY\nimport numpy\n'
    var_36 = module_0.file_contents(var_35, var_34)
    var_37 = True
    var_38 = module_1.Config()
    var_39 = 'import os\n'
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = var_40.verbose_output
    var_42 = len(var_41)
    var_43 = module_1.Config()
    var_44 = 'import os as os\n'
    var_45 = module_0.file_contents(var_44, var_43)
    var_46 = module_1.Config()
    var_47 = 'import module as mod  # comment\n'
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = '# noqa'
    var_50 = [var_49]
    var_51 = module_1.Config()
    var_52 = '# noqa\nimport os\n'
    var_53 = module_0.file_contents(var_52, var_51)
    var_54 = 'import os\r\nimport sys\r\n'
    var_55 = module_0.file_contents(var_54)
    var_56 = 'CUSTOM'
    var_57 = [var_56]
    var_58 = module_1.Config()
    var_59 = 'import os\n'
    var_60 = module_0.file_contents(var_59, var_58)
    var_61 = module_1.Config()
    var_62 = 'from module import func  # comment\n'
    var_63 = module_0.file_contents(var_62, var_61)



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import collections as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # system module\nimport sys  # system\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from very.long.module.name import (\\\n    function1,\\\n    function2\\\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (a, b, c,)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = True
    var_15 = module_1.Config()
    var_16 = 'from module import a, b, c\n'
    var_17 = module_0.file_contents(var_16, var_15)
    var_18 = ''
    var_19 = module_0.file_contents(var_18)
    var_20 = module_2.OrderedDict()
    var_21 = "def foo():\n    return 'bar'\n"
    var_22 = module_0.file_contents(var_21)
    var_23 = var_22.imports
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = 'from module import (\\\n    a,  # comment for a\\\n    b  # comment for b\\\n)\n'
    var_26 = module_0.file_contents(var_25)
    var_27 = '# This is a comment\n# Another comment\nimport os\n'
    var_28 = module_0.file_contents(var_27)
    var_29 = 'os'
    var_30 = 'straight'
    var_31 = 'above'
    var_32 = var_28.categorized_comments[var_31][var_30][var_29]
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = 'import os\r\nimport sys\r\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = module_1.Config()
    var_37 = 'import os\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = var_38.verbose_output
    var_40 = len(var_39)
    var_41 = module_1.Config()
    var_42 = "print('hello')\nimport os\n"
    var_43 = module_0.file_contents(var_42, var_41)
    var_44 = 'from module import (  # isort:skip\n    a,\n    b,\n)\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = var_45.imports
    var_47 = len(var_46)
    assert var_47 == 0
    var_48 = 'import os; import sys\n'
    var_49 = module_0.file_contents(var_48)
    var_50 = 'from cython cimport something\n'
    var_51 = module_0.file_contents(var_50)
    var_52 = module_1.Config()
    var_53 = 'import os as os\n'
    var_54 = module_0.file_contents(var_53, var_52)



# Parsed testcases at query #6
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os.path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..parent import child'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = "print('hello')"
    var_13 = module_0.import_type(var_12)
    assert var_13 is None
    var_14 = 'def function():'
    var_15 = module_0.import_type(var_14)
    assert var_15 is None
    var_16 = '# comment'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = 'import os  # noqa'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'from os import path  # NOQA'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # noqa: F401'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'import os  # isort:skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'from os import path  # isort: skip'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'import os  # isort:split'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = 'import'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = 'from'
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = ''
    var_35 = module_0.import_type(var_34)
    assert var_35 is None
    var_36 = '   '
    var_37 = module_0.import_type(var_36)
    assert var_37 is None
    var_38 = '  import os'
    var_39 = module_0.import_type(var_38)
    assert var_39 == 'straight'
    var_40 = '\tfrom os import path'
    var_41 = module_0.import_type(var_40)
    assert var_41 == 'from'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'import os'
    var_1 = 'cimport numpy'
    var_2 = 'import os.path'
    var_3 = '  import os'
    var_4 = 'from os import path'
    var_5 = 'from . import module'
    var_6 = 'from ..package import something'
    var_7 = '  from os import path'
    var_8 = 'import os  # noqa'
    var_9 = 'from os import path  # NOQA'
    var_10 = 'import os  # noqa: F401'
    var_11 = 'import os  # isort:skip'
    var_12 = 'from os import path  # isort: skip'
    var_13 = 'import os  # isort:split'
    var_14 = "print('hello')"
    var_15 = 'def function():'
    var_16 = '# This is a comment'
    var_17 = ''
    var_18 = '    '
    var_19 = 'IMPORT os'
    var_20 = 'FROM os IMPORT path'
    var_21 = 'Import os'
    var_22 = 'From os import path'



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
    var_4 = 'from os import path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'from'
    var_6 = "print('hello')"
    var_7 = module_0.import_type(var_6)
    assert var_7 is None
    var_8 = 'def function():'
    var_9 = module_0.import_type(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = module_0.import_type(var_10)
    assert var_11 is None
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os  # noqa'
    var_15 = module_0.import_type(var_14, var_13)
    assert var_15 is None
    var_16 = 'import os  # NOQA'
    var_17 = module_0.import_type(var_16, var_13)
    assert var_17 is None
    var_18 = 'from os import path  # noqa'
    var_19 = module_0.import_type(var_18, var_13)
    assert var_19 is None
    var_20 = False
    var_21 = module_1.Config()
    var_22 = module_0.import_type(var_14, var_21)
    assert var_22 == 'straight'
    var_23 = module_0.import_type(var_18, var_21)
    assert var_23 == 'from'
    var_24 = 'import os  # isort:skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort: skip'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'from os import path  # isort:skip'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = 'import os  # isort:split'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = 'import os  '
    var_33 = module_0.import_type(var_32)
    assert var_33 == 'straight'
    var_34 = 'from os import path  '
    var_35 = module_0.import_type(var_34)
    assert var_35 == 'from'
    var_36 = '\timport os'
    var_37 = module_0.import_type(var_36)
    assert var_37 == 'straight'
    var_38 = '\tfrom os import path'
    var_39 = module_0.import_type(var_38)
    assert var_39 == 'from'
    var_40 = 'important'
    var_41 = module_0.import_type(var_40)
    assert var_41 is None
    var_42 = 'fromage'
    var_43 = module_0.import_type(var_42)
    assert var_43 is None
    var_44 = 'import'
    var_45 = module_0.import_type(var_44)
    assert var_45 is None
    var_46 = 'from'
    var_47 = module_0.import_type(var_46)
    assert var_47 is None



# Parsed testcases at query #9
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # system module\nimport sys  # system module\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (\n    item1,\n    item2,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'FUTURE'
    var_13 = 'STDLIB'
    var_14 = 'THIRDPARTY'
    var_15 = 'FIRSTPARTY'
    var_16 = [var_12, var_13, var_14, var_15]
    var_17 = module_1.Config()
    var_18 = 'from __future__ import annotations\nimport os\nimport numpy\n'
    var_19 = module_0.file_contents(var_18, var_17)
    var_20 = ''
    var_21 = module_0.file_contents(var_20)
    var_22 = var_21.imports
    var_23 = len(var_22)
    var_24 = '# This is a comment\n# Another comment\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'from module import (\n    item1,  # comment1\n    item2,  # comment2\n)\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = True
    var_33 = module_1.Config()
    var_34 = 'import os\n'
    var_35 = module_0.file_contents(var_34, var_33)
    var_36 = var_35.verbose_output
    var_37 = len(var_36)
    var_38 = 'tests'
    var_39 = [var_38]
    var_40 = module_1.Config()
    var_41 = 'import os\nimport pytest\n'
    var_42 = module_0.file_contents(var_41, var_40)
    var_43 = '# Above comment\nimport os\n'
    var_44 = module_0.file_contents(var_43)
    var_45 = 'import os; import sys\n'
    var_46 = module_0.file_contents(var_45)
    var_47 = 'from module import item1, \\\n    item2, \\\n    item3\n'
    var_48 = module_0.file_contents(var_47)



# Parsed testcases at query #10
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
    var_6 = 'not an import'
    var_7 = module_0.import_type(var_6)
    assert var_7 is None
    var_8 = ''
    var_9 = module_0.import_type(var_8)
    assert var_9 is None
    var_10 = '   '
    var_11 = module_0.import_type(var_10)
    assert var_11 is None
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os  # noqa'
    var_15 = module_0.import_type(var_14, var_13)
    assert var_15 is None
    var_16 = 'import os  # NOQA'
    var_17 = module_0.import_type(var_16, var_13)
    assert var_17 is None
    var_18 = 'import os  # noqa: F401'
    var_19 = module_0.import_type(var_18, var_13)
    assert var_19 is None
    var_20 = module_0.import_type(var_0, var_13)
    assert var_20 == 'straight'
    var_21 = False
    var_22 = module_1.Config()
    var_23 = module_0.import_type(var_14, var_22)
    assert var_23 == 'straight'
    var_24 = 'import os  # isort:skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort: skip'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'import os  # isort:split'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = 'from os import path  # isort:skip'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = '  import os'
    var_33 = module_0.import_type(var_32)
    assert var_33 == 'straight'
    var_34 = '\timport os'
    var_35 = module_0.import_type(var_34)
    assert var_35 == 'straight'
    var_36 = '  from os import path'
    var_37 = module_0.import_type(var_36)
    assert var_37 == 'from'
    var_38 = 'import*'
    var_39 = module_0.import_type(var_38)
    assert var_39 is None
    var_40 = 'from.import x'
    var_41 = module_0.import_type(var_40)
    assert var_41 is None
    var_42 = 'import os.path'
    var_43 = module_0.import_type(var_42)
    assert var_43 == 'straight'
    var_44 = 'from . import module'
    var_45 = module_0.import_type(var_44)
    assert var_45 == 'from'
    var_46 = 'from .. import module'
    var_47 = module_0.import_type(var_46)
    assert var_47 == 'from'



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os.path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = '  import os'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'straight'
    var_8 = '\timport os'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'straight'
    var_10 = 'from os import path'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'from . import module'
    var_13 = module_0.import_type(var_12)
    assert var_13 == 'from'
    var_14 = '  from os import path'
    var_15 = module_0.import_type(var_14)
    assert var_15 == 'from'
    var_16 = '\tfrom os import path'
    var_17 = module_0.import_type(var_16)
    assert var_17 == 'from'
    var_18 = "print('hello')"
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'def function():'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = ''
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = '   '
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # noqa'
    var_27 = 'import os  # NOQA'
    var_28 = 'from os import path  # noqa'
    var_29 = 'import os  # noqa: F401'
    var_30 = 'import os  # isort:skip'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = 'from os import path  # isort: skip'
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = 'import os  # isort:split'
    var_35 = module_0.import_type(var_34)
    assert var_35 is None



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os.path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = '  import os'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'straight'
    var_8 = 'from os import path'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from . import module'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = '  from os import path'
    var_13 = module_0.import_type(var_12)
    assert var_13 == 'from'
    var_14 = "print('hello')"
    var_15 = module_0.import_type(var_14)
    assert var_15 is None
    var_16 = 'def function():'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = ''
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = '# comment'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # noqa'
    var_23 = 'from os import path  # NOQA'
    var_24 = 'import os  # noqa: F401'
    var_25 = 'import os  # isort:skip'
    var_26 = module_0.import_type(var_25)
    assert var_26 is None
    var_27 = 'from os import path  # isort: skip'
    var_28 = module_0.import_type(var_27)
    assert var_28 is None
    var_29 = 'import os  # isort:split'
    var_30 = module_0.import_type(var_29)
    assert var_30 is None



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "print('hello')"
    var_6 = ''
    var_7 = module_0.skip_line(var_5, var_6, var_2, var_3)
    var_8 = 'print("world")'
    var_9 = ''
    var_10 = module_0.skip_line(var_8, var_9, var_2, var_3)
    var_11 = "print('''hello''')"
    var_12 = ''
    var_13 = module_0.skip_line(var_11, var_12, var_2, var_3)
    var_14 = 'print("""world""")'
    var_15 = ''
    var_16 = module_0.skip_line(var_14, var_15, var_2, var_3)
    var_17 = 'some text'
    var_18 = "'"
    var_19 = module_0.skip_line(var_17, var_18, var_2, var_3)
    var_20 = "print('it\\'s me')"
    var_21 = ''
    var_22 = module_0.skip_line(var_20, var_21, var_2, var_3)
    var_23 = "world'"
    var_24 = "'"
    var_25 = module_0.skip_line(var_23, var_24, var_2, var_3)
    var_26 = "import os; print('test')"
    var_27 = module_0.skip_line(var_26, var_24, var_2, var_3)
    var_28 = 'x = 1; y = 2'
    var_29 = module_0.skip_line(var_28, var_24, var_2, var_3)
    var_30 = "'text' # comment"
    var_31 = ''
    var_32 = module_0.skip_line(var_30, var_31, var_2, var_3)
    var_33 = '"hello" + "world"'
    var_34 = ''
    var_35 = module_0.skip_line(var_33, var_34, var_2, var_3)
    var_36 = ''
    var_37 = module_0.skip_line(var_36, var_34, var_2, var_3)
    var_38 = '# just a comment'
    var_39 = module_0.skip_line(var_38, var_34, var_2, var_3)



# Parsed testcases at query #14
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import collections as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (\n    func1,\n    func2,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'testmodule'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'import testmodule\nimport os\n'
    var_16 = module_0.file_contents(var_15, var_14)
    var_17 = '# isort:imports-stdlib\nimport os\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = ''
    var_20 = module_0.file_contents(var_19)
    var_21 = module_2.OrderedDict()
    var_22 = "def foo():\n    return 'bar'\n"
    var_23 = module_0.file_contents(var_22)
    var_24 = module_2.OrderedDict()
    var_25 = 'import os\n\ndef foo():\n    import sys\n    return sys.version\n'
    var_26 = module_0.file_contents(var_25)
    var_27 = True
    var_28 = module_1.Config()
    var_29 = 'import os\n'
    var_30 = module_0.file_contents(var_29, var_28)
    var_31 = var_30.verbose_output
    var_32 = len(var_31)
    var_33 = 'import os\r\nimport sys\r\n'
    var_34 = module_0.file_contents(var_33)
    var_35 = '# This is a comment\nimport os\n'
    var_36 = module_0.file_contents(var_35)
    var_37 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_38 = module_0.file_contents(var_37)
    var_39 = module_1.Config()
    var_40 = 'import os as os\n'
    var_41 = module_0.file_contents(var_40, var_39)
    var_42 = 'from very.long.module.name import \\\n    function1, function2\n'
    var_43 = module_0.file_contents(var_42)
    var_44 = 'import os; import sys\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = module_1.Config()
    var_47 = "print('hello')\nimport os\n"
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = 'import os  # isort:skip\nimport sys\n'
    var_50 = module_0.file_contents(var_49)
    var_51 = 'FIRSTPARTY'
    var_52 = [var_51]
    var_53 = module_1.Config()
    var_54 = 'import os\n'
    var_55 = module_0.file_contents(var_54, var_53)
    var_56 = module_1.Config()
    var_57 = 'from module import func as f  # comment\n'
    var_58 = module_0.file_contents(var_57, var_56)



# Parsed testcases at query #15
#--------------------------


import isort.parse as module_0
import collections as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (\n    func1,\n    func2,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = ''
    var_15 = module_0.file_contents(var_14)
    var_16 = 'FUTURE'
    var_17 = 'straight'
    var_18 = 'from'
    var_19 = module_1.OrderedDict()
    var_20 = module_1.OrderedDict()
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = (var_16, var_21)
    var_23 = 'STDLIB'
    var_24 = module_1.OrderedDict()
    var_25 = module_1.OrderedDict()
    var_26 = {var_17: var_24, var_18: var_25}
    var_27 = (var_23, var_26)
    var_28 = 'THIRDPARTY'
    var_29 = module_1.OrderedDict()
    var_30 = module_1.OrderedDict()
    var_31 = {var_17: var_29, var_18: var_30}
    var_32 = (var_28, var_31)
    var_33 = 'FIRSTPARTY'
    var_34 = module_1.OrderedDict()
    var_35 = module_1.OrderedDict()
    var_36 = {var_17: var_34, var_18: var_35}
    var_37 = (var_33, var_36)
    var_38 = 'LOCALFOLDER'
    var_39 = module_1.OrderedDict()
    var_40 = module_1.OrderedDict()
    var_41 = {var_17: var_39, var_18: var_40}
    var_42 = (var_38, var_41)
    var_43 = [var_22, var_27, var_32, var_37, var_42]
    var_44 = 'tests'
    var_45 = [var_44]
    var_46 = module_2.Config()
    var_47 = 'import pytest\nimport mymodule\n'
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = True
    var_50 = module_2.Config()
    var_51 = 'import os\n'
    var_52 = module_0.file_contents(var_51, var_50)
    var_53 = var_52.verbose_output
    var_54 = len(var_53)
    var_55 = 'import os\r\nimport sys\r\n'
    var_56 = module_0.file_contents(var_55)
    var_57 = "print('hello')\nimport os\n"
    var_58 = module_0.file_contents(var_57)
    var_59 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_60 = module_0.file_contents(var_59)
    var_61 = 'import os  # isort:skip\nimport sys\n'
    var_62 = module_0.file_contents(var_61)
    var_63 = '# Above comment\nimport os\n'
    var_64 = module_0.file_contents(var_63)



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from very.long.module.name import (\n    first_thing,\n    second_thing,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'tests'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'import pytest\nimport mymodule\n'
    var_16 = module_0.file_contents(var_15, var_14)
    var_17 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = ''
    var_20 = module_0.file_contents(var_19)
    var_21 = 'straight'
    var_22 = 'STDLIB'
    var_23 = var_20.imports[var_22][var_21]
    var_24 = len(var_23)
    assert var_24 == 0
    var_25 = 'from'
    var_26 = var_20.imports[var_22][var_25]
    var_27 = len(var_26)
    assert var_27 == 0
    var_28 = "def foo():\n    return 'bar'\n"
    var_29 = module_0.file_contents(var_28)
    var_30 = var_29.imports[var_22][var_21]
    var_31 = len(var_30)
    assert var_31 == 0
    var_32 = True
    var_33 = module_1.Config()
    var_34 = 'import os\n'
    var_35 = module_0.file_contents(var_34, var_33)
    var_36 = var_35.verbose_output
    var_37 = len(var_36)
    var_38 = 'from typing import (\n    List,  # comment for List\n    Dict,  # comment for Dict\n)\n'
    var_39 = module_0.file_contents(var_38)
    var_40 = 'import os  # isort:skip\nimport sys\n'
    var_41 = module_0.file_contents(var_40)
    var_42 = 'import os\r\nimport sys\r\n'
    var_43 = module_0.file_contents(var_42)
    var_44 = '# This is a comment\nimport os\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = module_1.Config()
    var_47 = 'import numpy as np  # comment\n'
    var_48 = module_0.file_contents(var_47, var_46)



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "x = 'import os'"
    var_6 = ()
    var_7 = module_0.skip_line(var_5, var_1, var_2, var_6)
    var_8 = 'x = "import os"'
    var_9 = ()
    var_10 = module_0.skip_line(var_8, var_1, var_2, var_9)
    var_11 = "x = '''import os'''"
    var_12 = ()
    var_13 = module_0.skip_line(var_11, var_1, var_2, var_12)
    var_14 = 'x = """import os"""'
    var_15 = ()
    var_16 = module_0.skip_line(var_14, var_1, var_2, var_15)
    var_17 = "'import os' + 'test'"
    var_18 = "'"
    var_19 = ()
    var_20 = module_0.skip_line(var_17, var_18, var_2, var_19)
    var_21 = "x = 'it\\'s import'"
    var_22 = ()
    var_23 = module_0.skip_line(var_21, var_1, var_2, var_22)
    var_24 = 'x = 1; import os'
    var_25 = ()
    var_26 = True
    var_27 = module_0.skip_line(var_24, var_1, var_2, var_25, var_26)
    var_28 = 'import os; import sys'
    var_29 = ()
    var_30 = module_0.skip_line(var_28, var_1, var_2, var_29, var_26)
    var_31 = "x = 'import' # comment"
    var_32 = ()
    var_33 = module_0.skip_line(var_31, var_1, var_2, var_32)
    var_34 = 'continued line'
    var_35 = ()
    var_36 = module_0.skip_line(var_34, var_18, var_2, var_35)
    var_37 = "'start' + 'end'"
    var_38 = ()
    var_39 = module_0.skip_line(var_37, var_1, var_2, var_38)
    var_40 = 'x = 1; y = 2'
    var_41 = ()
    var_42 = False
    var_43 = module_0.skip_line(var_40, var_1, var_2, var_41, var_42)
    var_44 = ()
    var_45 = module_0.skip_line(var_1, var_1, var_42, var_44)
    var_46 = '# import os'
    var_47 = ()
    var_48 = module_0.skip_line(var_46, var_1, var_42, var_47)



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.Config()
    var_2 = module_1.file_contents(var_0, var_1)
    var_3 = 'from collections import defaultdict, OrderedDict\n'
    var_4 = module_1.file_contents(var_3, var_1)
    var_5 = 'import os  # system module\nimport sys  # system module\n'
    var_6 = module_1.file_contents(var_5, var_1)
    var_7 = 'from very.long.package.name import (\\\n    function1,\\\n    function2\\\n)\n'
    var_8 = module_1.file_contents(var_7, var_1)
    var_9 = 'import numpy as np\nimport pandas as pd\n'
    var_10 = module_1.file_contents(var_9, var_1)
    var_11 = 'from module import (\\\n    item1,\\\n    item2,\\\n)\n'
    var_12 = module_1.file_contents(var_11, var_1)
    var_13 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_14 = module_1.file_contents(var_13, var_1)
    var_15 = ''
    var_16 = module_1.file_contents(var_15, var_1)
    var_17 = var_16.imports
    var_18 = len(var_17)
    var_19 = "def foo():\n    return 'bar'\n"
    var_20 = module_1.file_contents(var_19, var_1)
    var_21 = var_20.lines_without_imports
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'test_module'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = 'import test_module\nimport os\n'
    var_27 = module_1.file_contents(var_26, var_25)
    var_28 = True
    var_29 = module_0.Config()
    var_30 = 'import os\n'
    var_31 = module_1.file_contents(var_30, var_29)
    var_32 = var_31.verbose_output
    var_33 = len(var_32)
    var_34 = 'import os\r\nimport sys\r\n'
    var_35 = module_1.file_contents(var_34, var_29)
    var_36 = 'from module import (\\\n    item1,  # comment1\\\n    item2  # comment2\\\n)\n'
    var_37 = module_1.file_contents(var_36, var_29)
    var_38 = 'import os  # isort:skip\nimport sys\n'
    var_39 = module_1.file_contents(var_38, var_29)
    var_40 = module_0.Config()
    var_41 = "print('hello')\nimport os\n"
    var_42 = module_1.file_contents(var_41, var_40)
    var_43 = 'import os; import sys\n'
    var_44 = module_1.file_contents(var_43, var_40)
    var_45 = module_0.Config()
    var_46 = 'import os as os\n'
    var_47 = module_1.file_contents(var_46, var_45)
    var_48 = module_0.Config()
    var_49 = 'import os as os_system\n# comment\n'
    var_50 = module_1.file_contents(var_49, var_48)



# Parsed testcases at query #19
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import collections as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = 'sys'
    var_6 = (var_5, var_3)
    var_7 = [var_4, var_6]
    var_8 = 'from collections import defaultdict, OrderedDict\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'defaultdict'
    var_11 = (var_10, var_3)
    var_12 = 'OrderedDict'
    var_13 = (var_12, var_3)
    var_14 = [var_11, var_13]
    var_15 = 'import os  # comment\nimport sys\n'
    var_16 = module_0.file_contents(var_15)
    var_17 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = 'from very.long.module.name import (\n    function1,\n    function2,\n)\n'
    var_20 = module_0.file_contents(var_19)
    var_21 = 'import numpy as np\nimport pandas as pd\n'
    var_22 = module_0.file_contents(var_21)
    var_23 = 'tests'
    var_24 = [var_23]
    var_25 = module_1.Config()
    var_26 = 'import pytest\nimport os\n'
    var_27 = module_0.file_contents(var_26, var_25)
    var_28 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = '# This is a comment\nimport os\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = module_1.Config()
    var_35 = 'import os\n'
    var_36 = module_0.file_contents(var_35, var_34)
    var_37 = var_36.verbose_output
    var_38 = len(var_37)
    var_39 = ''
    var_40 = module_0.file_contents(var_39)
    var_41 = module_2.OrderedDict()
    var_42 = 'import os\r\nimport sys\r\n'
    var_43 = module_0.file_contents(var_42)
    var_44 = 'import os  # isort:skip\nimport sys\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = module_1.Config()
    var_47 = "print('hello')\nimport os\n"
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = module_1.Config()
    var_50 = 'import numpy as np  # comment\n'
    var_51 = module_0.file_contents(var_50, var_49)
    var_52 = module_1.Config()
    var_53 = 'import os as os\n'
    var_54 = module_0.file_contents(var_53, var_52)
    var_55 = 'STDLIB'
    var_56 = 'THIRDPARTY'
    var_57 = [var_55, var_56]
    var_58 = module_1.Config()
    var_59 = 'import unknown_module\n'
    var_60 = module_0.file_contents(var_59, var_58)
    var_61 = module_1.Config()
    var_62 = 'from typing import List  # comment\n'
    var_63 = module_0.file_contents(var_62, var_61)
    var_64 = 'import os; import sys\n'
    var_65 = module_0.file_contents(var_64)
    var_66 = 'from libc cimport math\n'
    var_67 = module_0.file_contents(var_66)
    var_68 = 'from very.long.module \\\n    import function\n'
    var_69 = module_0.file_contents(var_68)
    var_70 = '# noqa'
    var_71 = [var_70]
    var_72 = module_1.Config()
    var_73 = '# noqa\nimport os\n'
    var_74 = module_0.file_contents(var_73, var_72)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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
    var_4 = 'from x import y'
    var_5 = module_0.strip_syntax(var_4)
    assert var_5 == 'x y'
    var_6 = 'from x import y, z'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'x y z'
    var_8 = 'cimport numpy'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'numpy'
    var_10 = 'from x cimport y'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == 'x y'
    var_12 = 'import os\\'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == 'os'
    var_14 = 'import os, \\'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == 'os'
    var_16 = 'import os\\\n    sys'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == 'os sys'
    var_18 = 'import (os)'
    var_19 = module_0.strip_syntax(var_18)
    assert var_19 == 'os'
    var_20 = 'import (os, sys)'
    var_21 = module_0.strip_syntax(var_20)
    assert var_21 == 'os sys'
    var_22 = 'from x import (y, z)'
    var_23 = module_0.strip_syntax(var_22)
    assert var_23 == 'x y z'
    var_24 = 'import os,sys'
    var_25 = module_0.strip_syntax(var_24)
    assert var_25 == 'os sys'
    var_26 = 'from x import y,z'
    var_27 = module_0.strip_syntax(var_26)
    assert var_27 == 'x y z'
    var_28 = 'from x import { y }'
    var_29 = module_0.strip_syntax(var_28)
    assert var_29 == 'x {|y|}'
    var_30 = 'from x import { y, z }'
    var_31 = module_0.strip_syntax(var_30)
    assert var_31 == 'x {|y z|}'
    var_32 = '_import os'
    var_33 = module_0.strip_syntax(var_32)
    assert var_33 == '_import os'
    var_34 = '_cimport numpy'
    var_35 = module_0.strip_syntax(var_34)
    assert var_35 == '_cimport numpy'
    var_36 = 'from x _import y'
    var_37 = module_0.strip_syntax(var_36)
    assert var_37 == 'x _import y'
    var_38 = 'from x _cimport y'
    var_39 = module_0.strip_syntax(var_38)
    assert var_39 == 'x _cimport y'
    var_40 = 'from module import (func1, func2, func3)'
    var_41 = module_0.strip_syntax(var_40)
    assert var_41 == 'module func1 func2 func3'
    var_42 = 'import os, sys, \\\n    math'
    var_43 = module_0.strip_syntax(var_42)
    assert var_43 == 'os sys math'
    var_44 = 'from a.b import c as d, e as f'
    var_45 = module_0.strip_syntax(var_44)
    assert var_45 == 'a.b c as d e as f'
    var_46 = ''
    var_47 = module_0.strip_syntax(var_46)
    assert var_47 == ''
    var_48 = '   '
    var_49 = module_0.strip_syntax(var_48)
    assert var_49 == ''
    var_50 = 'import  os,  sys'
    var_51 = module_0.strip_syntax(var_50)
    assert var_51 == 'os sys'
    var_52 = 'from  x  import  y'
    var_53 = module_0.strip_syntax(var_52)
    assert var_53 == 'x y'



# Parsed testcases at query #2
#--------------------------


import isort.parse as module_0
import collections as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from sklearn.ensemble import RandomForestClassifier as RFC\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from module import (\n    item1,\n    item2,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'import os\r\nimport sys\r\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = module_1.OrderedDict()
    var_19 = 'tests'
    var_20 = [var_19]
    var_21 = module_2.Config()
    var_22 = 'import pytest\nimport mymodule\n'
    var_23 = module_0.file_contents(var_22, var_21)
    var_24 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_25 = module_0.file_contents(var_24)
    var_26 = True
    var_27 = module_2.Config()
    var_28 = 'import os\n'
    var_29 = module_0.file_contents(var_28, var_27)
    var_30 = var_29.verbose_output
    var_31 = len(var_30)
    var_32 = 'import os  # isort:skip\nimport sys\n'
    var_33 = module_0.file_contents(var_32)
    var_34 = 'from very.long.module.name import \\\n    function1, \\\n    function2\n'
    var_35 = module_0.file_contents(var_34)
    var_36 = module_2.Config()
    var_37 = 'from module import submodule as sm\n# comment\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = '# Above comment\nimport os\n'
    var_40 = module_0.file_contents(var_39)
    var_41 = 'from module import (\n    item1,  # nested comment\n    item2,\n)\n'
    var_42 = module_0.file_contents(var_41)
    var_43 = 'import os; import sys\n'
    var_44 = module_0.file_contents(var_43)
    var_45 = 'from cython cimport function\n'
    var_46 = module_0.file_contents(var_45)
    var_47 = False
    var_48 = module_2.Config()
    var_49 = "print('hello')\nimport os\n"
    var_50 = module_0.file_contents(var_49, var_48)
    var_51 = module_2.Config()
    var_52 = "print('hello')\nimport os\n"
    var_53 = module_0.file_contents(var_52, var_51)



# Parsed testcases at query #3
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from very.long.module.name import (\n    first_thing,\n    second_thing,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'tests'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'import pytest\nimport mymodule\n'
    var_16 = module_0.file_contents(var_15, var_14)
    var_17 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = ''
    var_20 = module_0.file_contents(var_19)
    var_21 = var_20.lines_without_imports
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = "def foo():\n    return 'bar'\n"
    var_24 = module_0.file_contents(var_23)
    var_25 = 'straight'
    var_26 = 'STDLIB'
    var_27 = var_24.imports[var_26][var_25]
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = var_24.lines_without_imports
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 'import os\r\nimport sys\r\n'
    var_32 = module_0.file_contents(var_31)
    var_33 = 'from module import (\n    thing1,  # comment1\n    thing2,  # comment2\n)\n'
    var_34 = module_0.file_contents(var_33)
    var_35 = '# Above comment\nimport os\n'
    var_36 = module_0.file_contents(var_35)
    var_37 = True
    var_38 = module_1.Config()
    var_39 = 'import os\n'
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = var_40.verbose_output
    var_42 = len(var_41)
    var_43 = 'import os  # isort:skip\nimport sys\n'
    var_44 = module_0.file_contents(var_43)
    var_45 = var_44.lines_without_imports
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = 'import os; import sys\n'
    var_48 = module_0.file_contents(var_47)
    var_49 = 'from module import thing1, \\\n    thing2, thing3\n'
    var_50 = module_0.file_contents(var_49)
    var_51 = 'from libc.stdio cimport printf\n'
    var_52 = module_0.file_contents(var_51)



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
    var_6 = 'from os.path import join, split'
    var_7 = module_0.strip_syntax(var_6)
    assert var_7 == 'os.path join split'
    var_8 = 'cimport numpy'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 'numpy'
    var_10 = 'from numpy cimport array'
    var_11 = module_0.strip_syntax(var_10)
    assert var_11 == 'numpy array'
    var_12 = 'import my_module'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == 'my_module'
    var_14 = 'from my_package import my_module'
    var_15 = module_0.strip_syntax(var_14)
    assert var_15 == 'my_package my_module'
    var_16 = 'import os, \\\n    sys'
    var_17 = module_0.strip_syntax(var_16)
    assert var_17 == 'os sys'
    var_18 = 'from os import (path, sys)'
    var_19 = module_0.strip_syntax(var_18)
    assert var_19 == 'os path sys'
    var_20 = 'import (os, sys)'
    var_21 = module_0.strip_syntax(var_20)
    assert var_21 == 'os sys'
    var_22 = 'import os,sys'
    var_23 = module_0.strip_syntax(var_22)
    assert var_23 == 'os sys'
    var_24 = 'import os , sys'
    var_25 = module_0.strip_syntax(var_24)
    assert var_25 == 'os sys'
    var_26 = 'from os import { path, sys }'
    var_27 = module_0.strip_syntax(var_26)
    assert var_27 == 'os {|path sys|}'
    var_28 = 'import { os, sys }'
    var_29 = module_0.strip_syntax(var_28)
    assert var_29 == '{|os sys|}'
    var_30 = 'from my.package import (func1, func2, \\\n    func3)'
    var_31 = module_0.strip_syntax(var_30)
    assert var_31 == 'my.package func1 func2 func3'
    var_32 = ''
    var_33 = module_0.strip_syntax(var_32)
    assert var_33 == ''
    var_34 = '   '
    var_35 = module_0.strip_syntax(var_34)
    assert var_35 == ''
    var_36 = 'import my_import'
    var_37 = module_0.strip_syntax(var_36)
    assert var_37 == 'my[[i]]'
    var_38 = 'from package import my_import'
    var_39 = module_0.strip_syntax(var_38)
    assert var_39 == 'package my[[i]]'
    var_40 = 'import my_cimport'
    var_41 = module_0.strip_syntax(var_40)
    assert var_41 == 'my[[ci]]'
    var_42 = 'from package import my_cimport'
    var_43 = module_0.strip_syntax(var_42)
    assert var_43 == 'package my[[ci]]'
    var_44 = 'from os.path import (join, split, \\\n    abspath)'
    var_45 = module_0.strip_syntax(var_44)
    assert var_45 == 'os.path join split abspath'



# Parsed testcases at query #5
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nfrom pandas import DataFrame as df\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (\n    item1,\n    item2,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'FIRSTPARTY'
    var_13 = 'THIRDPARTY'
    var_14 = 'STDLIB'
    var_15 = [var_12, var_13, var_14]
    var_16 = module_1.Config()
    var_17 = 'import mymodule\nimport numpy\nimport os\n'
    var_18 = module_0.file_contents(var_17, var_16)
    var_19 = '# isort:imports-stdlib\nimport os\n'
    var_20 = module_0.file_contents(var_19)
    var_21 = 'tests'
    var_22 = [var_21]
    var_23 = module_1.Config()
    var_24 = 'import pytest\nimport os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = True
    var_27 = module_1.Config()
    var_28 = 'import os\n'
    var_29 = module_0.file_contents(var_28, var_27)
    var_30 = var_29.verbose_output
    var_31 = len(var_30)
    var_32 = ''
    var_33 = module_0.file_contents(var_32)
    var_34 = var_33.imports
    var_35 = len(var_34)
    assert var_35 == 5
    var_36 = '# Just a comment\n# Another comment\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = var_37.lines_without_imports
    var_39 = len(var_38)
    assert var_39 == 2
    var_40 = 'import os\r\nimport sys\r\n'
    var_41 = module_0.file_contents(var_40)
    var_42 = False
    var_43 = module_1.Config()
    var_44 = "print('hello')\nimport os\n"
    var_45 = module_0.file_contents(var_44, var_43)
    var_46 = module_1.Config()
    var_47 = "print('hello')\nimport os\n"
    var_48 = module_0.file_contents(var_47, var_46)
    var_49 = 'from module import (\n    item1,  # comment1\n    item2,  # comment2\n)\n'
    var_50 = module_0.file_contents(var_49)
    var_51 = 'import os; import sys\n'
    var_52 = module_0.file_contents(var_51)
    var_53 = 'from very.long.module.name import \\\n    function1, function2\n'
    var_54 = module_0.file_contents(var_53)
    var_55 = module_1.Config()
    var_56 = 'import os as os\nfrom sys import exit as exit\n'
    var_57 = module_0.file_contents(var_56, var_55)
    var_58 = module_1.Config()
    var_59 = 'import pandas as pd  # data analysis\nimport numpy as np  # numerical\n'
    var_60 = module_0.file_contents(var_59, var_58)
    var_61 = module_1.Config()
    var_62 = '# Important comment\nimport os\n'
    var_63 = module_0.file_contents(var_62, var_61)
    var_64 = var_63.lines_without_imports
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = 'CUSTOM'
    var_67 = [var_66]
    var_68 = module_1.Config()
    var_69 = 'import os\n'
    var_70 = module_0.file_contents(var_69, var_68)
    var_71 = '# STDLIB'
    var_72 = '# THIRDPARTY'
    var_73 = [var_71, var_72]
    var_74 = module_1.Config()
    var_75 = '# STDLIB\nimport os\n# THIRDPARTY\nimport numpy\n'
    var_76 = module_0.file_contents(var_75, var_74)



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import os\nimport sys\n'
    var_2 = module_1.file_contents(var_1, var_0)
    var_3 = 'from collections import defaultdict, OrderedDict\n'
    var_4 = module_1.file_contents(var_3, var_0)
    var_5 = 'import os  # comment\nimport sys\n'
    var_6 = module_1.file_contents(var_5, var_0)
    var_7 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_8 = module_1.file_contents(var_7, var_0)
    var_9 = 'from very.long.package.name import (\n    first_thing,\n    second_thing,\n)\n'
    var_10 = module_1.file_contents(var_9, var_0)
    var_11 = 'import numpy as np\nimport pandas as pd\n'
    var_12 = module_1.file_contents(var_11, var_0)
    var_13 = 'from collections import defaultdict as dd, OrderedDict as od\n'
    var_14 = module_1.file_contents(var_13, var_0)
    var_15 = 'from typing import (\n    List,  # comment1\n    Dict,  # comment2\n)\n'
    var_16 = module_1.file_contents(var_15, var_0)
    var_17 = '# Above comment\nimport os\n'
    var_18 = module_1.file_contents(var_17, var_0)
    var_19 = 'tests'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = 'import os\nimport pytest\n'
    var_23 = module_1.file_contents(var_22, var_21)
    var_24 = '# isort:imports-stdlib\nimport os\n'
    var_25 = module_1.file_contents(var_24, var_21)
    var_26 = 'FIRSTPARTY'
    var_27 = [var_26]
    var_28 = module_0.Config()
    var_29 = 'import os\n'
    var_30 = module_1.file_contents(var_29, var_28)
    var_31 = True
    var_32 = module_0.Config()
    var_33 = 'import os\n'
    var_34 = module_1.file_contents(var_33, var_32)
    var_35 = var_34.verbose_output
    var_36 = len(var_35)
    var_37 = 'import os\r\nimport sys\r\n'
    var_38 = module_1.file_contents(var_37, var_32)
    var_39 = ''
    var_40 = module_1.file_contents(var_39, var_32)
    var_41 = 'straight'
    var_42 = var_40.imports[var_26][var_41]
    var_43 = len(var_42)
    assert var_43 == 0
    var_44 = 'from'
    var_45 = var_40.imports[var_26][var_44]
    var_46 = len(var_45)
    assert var_46 == 0
    var_47 = '# Just a comment\n# Another comment\n'
    var_48 = module_1.file_contents(var_47, var_32)
    var_49 = var_48.lines_without_imports
    var_50 = len(var_49)
    assert var_50 == 2
    var_51 = 'import os; import sys\n'
    var_52 = module_1.file_contents(var_51, var_32)
    var_53 = module_0.Config()
    var_54 = "print('hello')\nimport os\n"
    var_55 = module_1.file_contents(var_54, var_53)
    var_56 = '# noqa'
    var_57 = [var_56]
    var_58 = module_0.Config()
    var_59 = '# noqa\nimport os\n'
    var_60 = module_1.file_contents(var_59, var_58)
    var_61 = 'above'
    var_62 = var_60.categorized_comments[var_61][var_41]
    var_63 = 'os'
    var_64 = []
    var_65 = module_0.Config()
    var_66 = 'import os as os\n'
    var_67 = module_1.file_contents(var_66, var_65)
    var_68 = module_0.Config()
    var_69 = 'import os as os_system  # comment\n'
    var_70 = module_1.file_contents(var_69, var_68)



# Parsed testcases at query #7
#--------------------------


import isort.parse as module_0
import isort.settings as module_1
import collections as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # system module\nimport sys  # system module\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import (\n    func1,\n    func2,\n)\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = True
    var_13 = module_1.Config()
    var_14 = "print('hello')\nimport os\n"
    var_15 = module_0.file_contents(var_14, var_13)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = module_2.OrderedDict()
    var_19 = '# This is a comment\n# Another comment\n'
    var_20 = module_0.file_contents(var_19)
    var_21 = var_20.lines_without_imports
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = '# isort:imports-stdlib\nimport os\n'
    var_24 = module_0.file_contents(var_23)
    var_25 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_26 = module_0.file_contents(var_25)
    var_27 = 'from module import func1, \\\n    func2, \\\n    func3\n'
    var_28 = module_0.file_contents(var_27)
    var_29 = "import os\n\nprint('hello')\n\nimport sys\n"
    var_30 = module_0.file_contents(var_29)
    var_31 = var_30.lines_without_imports
    var_32 = len(var_31)
    assert var_32 == 3
    var_33 = '# STDLIB'
    var_34 = '# THIRDPARTY'
    var_35 = [var_33, var_34]
    var_36 = module_1.Config()
    var_37 = '# STDLIB\nimport os\n# THIRDPARTY\nimport numpy\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = module_1.Config()
    var_40 = 'import os\n'
    var_41 = module_0.file_contents(var_40, var_39)
    var_42 = var_41.verbose_output
    var_43 = len(var_42)
    var_44 = 'import os\r\nimport sys\r\n'
    var_45 = module_0.file_contents(var_44)
    var_46 = 'import os; import sys\n'
    var_47 = module_0.file_contents(var_46)
    var_48 = '# Comment above\nimport os\n'
    var_49 = module_0.file_contents(var_48)



# Parsed testcases at query #8
#--------------------------


import isort.parse as module_0
import collections as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from very.long.module.name import (\n    function1,\n    function2,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from collections import defaultdict as dd, OrderedDict as od\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from module import (\n    func1,\n    func2,\n)\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = 'FUTURE'
    var_19 = 'straight'
    var_20 = 'from'
    var_21 = module_1.OrderedDict()
    var_22 = module_1.OrderedDict()
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = (var_18, var_23)
    var_25 = 'STDLIB'
    var_26 = module_1.OrderedDict()
    var_27 = module_1.OrderedDict()
    var_28 = {var_19: var_26, var_20: var_27}
    var_29 = (var_25, var_28)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.OrderedDict()
    var_32 = module_1.OrderedDict()
    var_33 = {var_19: var_31, var_20: var_32}
    var_34 = (var_30, var_33)
    var_35 = 'FIRSTPARTY'
    var_36 = module_1.OrderedDict()
    var_37 = module_1.OrderedDict()
    var_38 = {var_19: var_36, var_20: var_37}
    var_39 = (var_35, var_38)
    var_40 = 'LOCALFOLDER'
    var_41 = module_1.OrderedDict()
    var_42 = module_1.OrderedDict()
    var_43 = {var_19: var_41, var_20: var_42}
    var_44 = (var_40, var_43)
    var_45 = [var_24, var_29, var_34, var_39, var_44]
    var_46 = "def foo():\n    return 'bar'\n"
    var_47 = module_0.file_contents(var_46)
    var_48 = 0
    var_49 = 'import os\n\ndef foo():\n    import sys\n    return sys.version\n'
    var_50 = module_0.file_contents(var_49)
    var_51 = 'import os\r\nimport sys\r\n'
    var_52 = module_0.file_contents(var_51)
    var_53 = True
    var_54 = module_2.Config()
    var_55 = 'import os\nimport sys\n'
    var_56 = module_0.file_contents(var_55, var_54)
    var_57 = var_56.verbose_output
    var_58 = len(var_57)
    var_59 = module_2.Config()
    var_60 = 'from module import func1, func2  # comment\n'
    var_61 = module_0.file_contents(var_60, var_59)
    var_62 = module_2.Config()
    var_63 = 'import os as os\nimport sys as system\n'
    var_64 = module_0.file_contents(var_63, var_62)
    var_65 = module_2.Config()
    var_66 = "print('hello')\nimport os\n"
    var_67 = module_0.file_contents(var_66, var_65)
    var_68 = 'import os; import sys\n'
    var_69 = module_0.file_contents(var_68)
    var_70 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_71 = module_0.file_contents(var_70)
    var_72 = 'nested'
    var_73 = var_71.categorized_comments[var_72]
    var_74 = 'module'
    var_75 = {}
    var_76 = var_71.categorized_comments[var_72]
    var_77 = {}
    var_78 = '# Above comment\nimport os\n'
    var_79 = module_0.file_contents(var_78)



# Parsed testcases at query #9
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
    var_6 = "print('hello')"
    var_7 = module_0.import_type(var_6)
    assert var_7 is None
    var_8 = 'def function():'
    var_9 = module_0.import_type(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = module_0.import_type(var_10)
    assert var_11 is None
    var_12 = True
    var_13 = module_1.Config()
    var_14 = 'import os  # noqa'
    var_15 = module_0.import_type(var_14, var_13)
    assert var_15 is None
    var_16 = 'from os import path  # NOQA'
    var_17 = module_0.import_type(var_16, var_13)
    assert var_17 is None
    var_18 = 'import os  # noqa: F401'
    var_19 = module_0.import_type(var_18, var_13)
    assert var_19 is None
    var_20 = 'import os  # other comment'
    var_21 = module_0.import_type(var_20, var_13)
    assert var_21 == 'straight'
    var_22 = 'import os  # isort:skip'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'from os import path  # isort: skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort:split'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = '\timport os'
    var_29 = module_0.import_type(var_28)
    assert var_29 == 'straight'
    var_30 = '  from os import path'
    var_31 = module_0.import_type(var_30)
    assert var_31 == 'from'
    var_32 = False
    var_33 = module_1.Config()
    var_34 = module_0.import_type(var_14, var_33)
    assert var_34 == 'straight'
    var_35 = module_0.import_type(var_16, var_33)
    assert var_35 == 'from'



# Parsed testcases at query #10
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "print('hello')"
    var_6 = "'"
    var_7 = module_0.skip_line(var_5, var_6, var_2, var_3)
    var_8 = "x = 'test'"
    var_9 = ''
    var_10 = module_0.skip_line(var_8, var_9, var_2, var_3)
    var_11 = 'y = "test"'
    var_12 = ''
    var_13 = module_0.skip_line(var_11, var_12, var_2, var_3)
    var_14 = "z = '''test'''"
    var_15 = ''
    var_16 = module_0.skip_line(var_14, var_15, var_2, var_3)
    var_17 = 'w = """test"""'
    var_18 = ''
    var_19 = module_0.skip_line(var_17, var_18, var_2, var_3)
    var_20 = "x = 'test\\'s'"
    var_21 = "'"
    var_22 = module_0.skip_line(var_20, var_21, var_2, var_3)
    var_23 = "end'"
    var_24 = "'"
    var_25 = module_0.skip_line(var_23, var_24, var_2, var_3)
    var_26 = "'test' # comment"
    var_27 = ''
    var_28 = module_0.skip_line(var_26, var_27, var_2, var_3)
    var_29 = "import os; print('hello')"
    var_30 = module_0.skip_line(var_29, var_27, var_2, var_3)
    var_31 = 'x = 1; y = 2'
    var_32 = module_0.skip_line(var_31, var_27, var_2, var_3)
    var_33 = "from os import path; print('test')"
    var_34 = module_0.skip_line(var_33, var_27, var_2, var_3)
    var_35 = '"test" + \'test\''
    var_36 = module_0.skip_line(var_35, var_27, var_2, var_3)
    var_37 = ''
    var_38 = module_0.skip_line(var_37, var_27, var_2, var_3)
    var_39 = '# This is a comment'
    var_40 = module_0.skip_line(var_39, var_27, var_2, var_3)
    var_41 = 'x = 1; y = 2'
    var_42 = False
    var_43 = module_0.skip_line(var_41, var_27, var_2, var_3, var_42)



# Parsed testcases at query #11
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\\\n    func1,\\\n    func2)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from pandas import DataFrame as df\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'from module import func1, func2,\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = '# isort:imports-stdlib\nimport os\n'
    var_15 = module_0.file_contents(var_14)
    var_16 = ''
    var_17 = module_0.file_contents(var_16)
    var_18 = var_17.imports
    var_19 = len(var_18)
    var_20 = 'def foo():\n    pass\n'
    var_21 = module_0.file_contents(var_20)
    var_22 = True
    var_23 = module_1.Config()
    var_24 = 'x = 1\nimport os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = 'from module import (  # comment1\n    func1,  # comment2\n    func2)\n'
    var_27 = module_0.file_contents(var_26)
    var_28 = 'import os  # isort:skip\nimport sys\n'
    var_29 = module_0.file_contents(var_28)
    var_30 = 'import os\r\nimport sys\r\n'
    var_31 = module_0.file_contents(var_30)
    var_32 = module_1.Config()
    var_33 = 'import os\n'
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = var_34.verbose_output
    var_36 = len(var_35)
    var_37 = 'test'
    var_38 = [var_37]
    var_39 = module_1.Config()
    var_40 = 'import os\n'
    var_41 = module_0.file_contents(var_40, var_39)
    var_42 = 'FIRST'
    var_43 = 'SECOND'
    var_44 = [var_42, var_43]
    var_45 = module_1.Config()
    var_46 = 'import os\n'
    var_47 = module_0.file_contents(var_46, var_45)
    var_48 = module_1.Config()
    var_49 = 'import os as os\n'
    var_50 = module_0.file_contents(var_49, var_48)
    var_51 = module_1.Config()
    var_52 = 'import os as operating_system\n'
    var_53 = module_0.file_contents(var_52, var_51)
    var_54 = module_1.Config()
    var_55 = '# comment\nimport os\n'
    var_56 = module_0.file_contents(var_55, var_54)
    var_57 = 'straight'
    var_58 = 'above'
    var_59 = var_56.categorized_comments[var_58][var_57]
    var_60 = 'os'
    var_61 = []
    var_62 = 'import os; import sys\n'
    var_63 = module_0.file_contents(var_62)
    var_64 = 'from cython cimport parallel\n'
    var_65 = module_0.file_contents(var_64)



# Parsed testcases at query #12
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict\nfrom typing import List, Dict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = '# This is a comment\nimport os  # inline comment\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from typing import (\n    List,\n    Dict,\n)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'from very.long.module.name import (\n    function1,\n    function2,\n)\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = 'tests'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'import pytest\nimport os\n'
    var_16 = module_0.file_contents(var_15, var_14)
    var_17 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = ''
    var_20 = module_0.file_contents(var_19)
    var_21 = var_20.imports
    var_22 = len(var_21)
    var_23 = "def foo():\n    return 'bar'\n"
    var_24 = module_0.file_contents(var_23)
    var_25 = var_24.lines_without_imports
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'import os\r\nimport sys\r\n'
    var_28 = module_0.file_contents(var_27)
    var_29 = 'from module import (\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_30 = module_0.file_contents(var_29)
    var_31 = '# Above comment\nimport os\n'
    var_32 = module_0.file_contents(var_31)
    var_33 = 'import os  # isort:skip\nimport sys\n'
    var_34 = module_0.file_contents(var_33)
    var_35 = True
    var_36 = module_1.Config()
    var_37 = 'import os\n'
    var_38 = module_0.file_contents(var_37, var_36)
    var_39 = var_38.verbose_output
    var_40 = len(var_39)
    var_41 = 'import os; import sys\n'
    var_42 = module_0.file_contents(var_41)
    var_43 = module_1.Config()
    var_44 = "print('hello')\nimport os\n"
    var_45 = module_0.file_contents(var_44, var_43)
    var_46 = module_1.Config()
    var_47 = 'import os as os\n'
    var_48 = module_0.file_contents(var_47, var_46)



# Parsed testcases at query #13
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'from collections import defaultdict, OrderedDict,\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'import os  # comment\nimport sys\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\nimport pandas as pd\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'tests'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'import pytest\nimport os\n'
    var_14 = module_0.file_contents(var_13, var_12)
    var_15 = '# isort:imports-thirdparty\nimport requests\n'
    var_16 = module_0.file_contents(var_15)
    var_17 = 'from very.long.package.name import (\\\n    function1,\\\n    function2)\n'
    var_18 = module_0.file_contents(var_17)
    var_19 = 'from module import (  # noqa\n    func1,  # comment1\n    func2,  # comment2\n)\n'
    var_20 = module_0.file_contents(var_19)
    var_21 = 'FIRSTPARTY'
    var_22 = [var_21]
    var_23 = module_1.Config()
    var_24 = 'import os\n'
    var_25 = module_0.file_contents(var_24, var_23)
    var_26 = True
    var_27 = module_1.Config()
    var_28 = 'import os\n'
    var_29 = module_0.file_contents(var_28, var_27)
    var_30 = var_29.verbose_output
    var_31 = len(var_30)
    var_32 = module_1.Config()
    var_33 = "print('hello')\nimport os\n"
    var_34 = module_0.file_contents(var_33, var_32)
    var_35 = module_1.Config()
    var_36 = '# comment\nimport os\n'
    var_37 = module_0.file_contents(var_36, var_35)
    var_38 = var_37.lines_without_imports
    var_39 = len(var_38)
    assert var_39 == 1
    var_40 = ''
    var_41 = module_0.file_contents(var_40)
    var_42 = var_41.imports
    var_43 = len(var_42)
    var_44 = "def foo():\n    return 'bar'\n"
    var_45 = module_0.file_contents(var_44)
    var_46 = var_45.lines_without_imports
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = '#!/usr/bin/env python\nimport os\n'
    var_49 = module_0.file_contents(var_48)
    var_50 = 'import os; import sys\n'
    var_51 = module_0.file_contents(var_50)
    var_52 = 'from libc.math cimport sin, cos\n'
    var_53 = module_0.file_contents(var_52)
    var_54 = module_1.Config()
    var_55 = 'import os as os\n'
    var_56 = module_0.file_contents(var_55, var_54)
    var_57 = module_1.Config()
    var_58 = 'import os as os_system\n# comment\n'
    var_59 = module_0.file_contents(var_58, var_57)
    var_60 = module_1.Config()
    var_61 = 'from module import func1, func2  # comment\n'
    var_62 = module_0.file_contents(var_61, var_60)
    var_63 = 'import os\r\nimport sys\r\n'
    var_64 = module_0.file_contents(var_63)
    var_65 = 'import os  # isort:skip\nimport sys\n'
    var_66 = module_0.file_contents(var_65)
    var_67 = var_66.lines_without_imports
    var_68 = len(var_67)
    assert var_68 == 2



# Parsed testcases at query #14
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
    var_4 = 'import os.path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from collections import defaultdict'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = True
    var_11 = module_1.Config()
    var_12 = 'import os  # noqa'
    var_13 = module_0.import_type(var_12, var_11)
    assert var_13 is None
    var_14 = 'import os  # NOQA'
    var_15 = module_0.import_type(var_14, var_11)
    assert var_15 is None
    var_16 = 'from os import path  # noqa'
    var_17 = module_0.import_type(var_16, var_11)
    assert var_17 is None
    var_18 = 'import os  # isort:skip'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'from os import path  # isort: skip'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # isort:split'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = "print('hello')"
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'def function():'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = ''
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = '    # comment'
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = module_1.Config()
    var_33 = module_0.import_type(var_12, var_32)
    assert var_33 == 'straight'
    var_34 = module_0.import_type(var_16, var_32)
    assert var_34 == 'from'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'import os'
    var_2 = module_1.import_type(var_1, var_0)
    assert var_2 == 'straight'
    var_3 = 'cimport numpy'
    var_4 = module_1.import_type(var_3, var_0)
    assert var_4 == 'straight'
    var_5 = 'import os.path'
    var_6 = module_1.import_type(var_5, var_0)
    assert var_6 == 'straight'
    var_7 = 'from os import path'
    var_8 = module_1.import_type(var_7, var_0)
    assert var_8 == 'from'
    var_9 = 'from . import module'
    var_10 = module_1.import_type(var_9, var_0)
    assert var_10 == 'from'
    var_11 = 'from ..package import something'
    var_12 = module_1.import_type(var_11, var_0)
    assert var_12 == 'from'
    var_13 = 'import os  # noqa'
    var_14 = module_1.import_type(var_13, var_0)
    assert var_14 is None
    var_15 = 'from os import path  # NOQA'
    var_16 = module_1.import_type(var_15, var_0)
    assert var_16 is None
    var_17 = 'import os  # noqa: F401'
    var_18 = module_1.import_type(var_17, var_0)
    assert var_18 is None
    var_19 = module_1.import_type(var_13, var_0)
    assert var_19 == 'straight'
    var_20 = module_1.import_type(var_15, var_0)
    assert var_20 == 'from'
    var_21 = 'import os  # isort:skip'
    var_22 = module_1.import_type(var_21, var_0)
    assert var_22 is None
    var_23 = 'from os import path  # isort: skip'
    var_24 = module_1.import_type(var_23, var_0)
    assert var_24 is None
    var_25 = 'import os  # isort:split'
    var_26 = module_1.import_type(var_25, var_0)
    assert var_26 is None
    var_27 = "print('hello')"
    var_28 = module_1.import_type(var_27, var_0)
    assert var_28 is None
    var_29 = 'def function():'
    var_30 = module_1.import_type(var_29, var_0)
    assert var_30 is None
    var_31 = '# This is a comment'
    var_32 = module_1.import_type(var_31, var_0)
    assert var_32 is None
    var_33 = ''
    var_34 = module_1.import_type(var_33, var_0)
    assert var_34 is None
    var_35 = '    '
    var_36 = module_1.import_type(var_35, var_0)
    assert var_36 is None
    var_37 = '    import os'
    var_38 = module_1.import_type(var_37, var_0)
    assert var_38 == 'straight'
    var_39 = '\tfrom os import path'
    var_40 = module_1.import_type(var_39, var_0)
    assert var_40 == 'from'
    var_41 = 'IMPORT os'
    var_42 = module_1.import_type(var_41, var_0)
    assert var_42 is None
    var_43 = 'FROM os import path'
    var_44 = module_1.import_type(var_43, var_0)
    assert var_44 is None
    var_45 = 'imported module'
    var_46 = module_1.import_type(var_45, var_0)
    assert var_46 is None
    var_47 = 'fromage cheese'
    var_48 = module_1.import_type(var_47, var_0)
    assert var_48 is None



# Parsed testcases at query #16
#--------------------------


import isort.parse as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from collections import defaultdict, OrderedDict\n'
    var_3 = module_0.file_contents(var_2)
    var_4 = 'import os  # comment\nimport sys\n'
    var_5 = module_0.file_contents(var_4)
    var_6 = 'from module import (\\\n    func1,\\\n    func2)\n'
    var_7 = module_0.file_contents(var_6)
    var_8 = 'import numpy as np\n'
    var_9 = module_0.file_contents(var_8)
    var_10 = 'from module import func1, func2,\n'
    var_11 = module_0.file_contents(var_10)
    var_12 = '# isort:imports-stdlib\nimport os\n'
    var_13 = module_0.file_contents(var_12)
    var_14 = 'STDLIB'
    var_15 = var_13.place_imports[var_14]
    var_16 = len(var_15)
    assert var_16 == 0
    var_17 = ''
    var_18 = module_0.file_contents(var_17)
    var_19 = var_18.lines_without_imports
    var_20 = len(var_19)
    assert var_20 == 0
    var_21 = 'def foo():\n    pass\n'
    var_22 = module_0.file_contents(var_21)
    var_23 = var_22.lines_without_imports
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'tests'
    var_26 = [var_25]
    var_27 = module_1.Config()
    var_28 = 'import pytest\nimport os\n'
    var_29 = module_0.file_contents(var_28, var_27)
    var_30 = True
    var_31 = module_1.Config()
    var_32 = 'import os\n'
    var_33 = module_0.file_contents(var_32, var_31)
    var_34 = var_33.verbose_output
    var_35 = len(var_34)
    var_36 = 'import os\r\nimport sys\r\n'
    var_37 = module_0.file_contents(var_36)
    var_38 = module_1.Config()
    var_39 = "print('hello')\nimport os\n"
    var_40 = module_0.file_contents(var_39, var_38)
    var_41 = 'from module import (\\\n    func1,  # comment1\\\n    func2  # comment2)\n'
    var_42 = module_0.file_contents(var_41)
    var_43 = 'import os  # isort:skip\nimport sys\n'
    var_44 = module_0.file_contents(var_43)
    var_45 = module_1.Config()
    var_46 = 'import module as mod  # comment\n'
    var_47 = module_0.file_contents(var_46, var_45)
    var_48 = module_1.Config()
    var_49 = 'import module as module\n'
    var_50 = module_0.file_contents(var_49, var_48)
    var_51 = module_1.Config()
    var_52 = 'from module import func1, func2  # comment\n'
    var_53 = module_0.file_contents(var_52, var_51)



# Parsed testcases at query #17
#--------------------------


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import os.path'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'from os import path'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'from'
    var_8 = 'from . import module'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'from'
    var_10 = 'from ..parent import child'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = "print('hello')"
    var_13 = module_0.import_type(var_12)
    assert var_13 is None
    var_14 = 'def function():'
    var_15 = module_0.import_type(var_14)
    assert var_15 is None
    var_16 = '# comment'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = 'import os  # noqa'
    var_19 = module_0.import_type(var_18)
    assert var_19 == 'straight'
    var_20 = 'from os import path  # NOQA'
    var_21 = module_0.import_type(var_20)
    assert var_21 == 'from'
    var_22 = 'import os  # isort:skip'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'from os import path  # isort: skip'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'import os  # isort:split'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'import os  '
    var_29 = module_0.import_type(var_28)
    assert var_29 == 'straight'
    var_30 = 'from os import path  '
    var_31 = module_0.import_type(var_30)
    assert var_31 == 'from'
    var_32 = ''
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = '\timport os'
    var_35 = module_0.import_type(var_34)
    assert var_35 == 'straight'
    var_36 = '\tfrom os import path'
    var_37 = module_0.import_type(var_36)
    assert var_37 == 'from'



