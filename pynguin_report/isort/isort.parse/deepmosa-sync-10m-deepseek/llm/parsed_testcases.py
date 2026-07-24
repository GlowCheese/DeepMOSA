####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['above']['straight']['os']
    var_5 = bool(var_1.categorized_comments['above']['straight']['os'] == ['# comment'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sep)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'sep'
    var_7 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'np'
    var_5 = bool('np' in var_1.as_map['straight']['numpy'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'numpy'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import numpy\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'numpy'
    var_8 = bool('numpy' in var_6.imports['numpy']['straight'])
    assert var_8 is True
    var_9 = 'os'
    var_10 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# stdlib'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# stdlib\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'os'
    var_8 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = '# isort:imports-stdlib'
    var_5 = bool('# isort:imports-stdlib' in var_1.import_placements)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.imports
    var_4 = len(var_3)
    assert var_4 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' not in var_1.imports['STDLIB']['straight'])
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as operating_system'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' not in var_5.as_map['straight'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'path'
    var_3 = bool('path' in var_1.categorized_comments['nested']['os'])
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'path'
    var_7 = bool('path' in var_5.categorized_comments['nested']['os'])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 14/29 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n# comment\nimport sys'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.out_lines
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = ''
    var_8 = '#'
    var_9 = '"""'
    var_10 = "'''"
    var_11 = 'isort:imports-'
    var_12 = 'isort: imports-'
    var_13 = var_1.treat_all_comments_as_code
    var_14 = var_1.treat_comments_as_code



# Parsed testcases at query #3
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sep)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'sep'
    var_7 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'np'
    var_5 = bool('np' in var_1.as_map['straight']['numpy'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from pandas import DataFrame as df\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'pandas'
    var_3 = bool('pandas' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'DataFrame'
    var_5 = bool('DataFrame' in var_1.imports['THIRDPARTY']['from']['pandas'])
    assert var_5 is True
    var_6 = 'df'
    var_7 = bool('df' in var_1.as_map['from']['pandas.DataFrame'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'pandas'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import pandas\nimport numpy\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'pandas'
    var_8 = bool('pandas' in var_6.imports['pandas']['straight'])
    assert var_8 is True
    var_9 = 'numpy'
    var_10 = bool('numpy' in var_6.imports['THIRDPARTY']['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# stdlib'
    var_1 = '# thirdparty'
    var_2 = [var_0, var_1]
    var_3 = 'section_comments'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '# stdlib\nimport os\n# thirdparty\nimport numpy\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = 'os'
    var_9 = bool('os' in var_7.imports['STDLIB']['straight'])
    assert var_9 is True
    var_10 = 'numpy'
    var_11 = bool('numpy' in var_7.imports['THIRDPARTY']['straight'])
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path,)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'straight'
    var_3 = 'STDLIB'
    var_4 = var_1.imports[var_3][var_2]
    var_5 = len(var_4)
    assert var_5 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pd\nimport pandas as pd2\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' in var_5.imports['THIRDPARTY']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pandas\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' not in var_5.as_map['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import unknown_module\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\n# another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, \\\n    sep\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'path'
    var_3 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_3 is True
    var_4 = 'sep'
    var_5 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc cimport math\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'libc'
    var_3 = bool('libc' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# special'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# special\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\\\n    sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = []
    var_3 = None
    var_4 = var_2 is not var_3
    var_5 = var_1 and var_4
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('Hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # isort:skip ('
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "'import os'"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "# comment\nprint('Hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "'''docstring'''\nprint('Hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "\n\nprint('Hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '# isort:imports-future\nimport os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import something  # comment'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['from']['module']
    var_5 = bool(var_3.imports['from']['module'] is not None)
    assert var_5 is True
    var_6 = var_3.categorized_comments['from']['module']
    var_7 = bool(var_3.categorized_comments['from']['module'] == ['# comment'])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_predicate_false. Retrieved 12/32 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = -1
    var_4 = 0
    var_5 = ''
    var_6 = ''
    var_7 = var_2.float_to_top
    var_8 = -1
    var_9 = var_3 == var_8
    var_10 = '#'
    var_11 = "'''"
    var_12 = '"""'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_empty_file. Retrieved 2/3 statements.
# Partially parsed test_file_contents_only_comments. Retrieved 2/3 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict, OrderedDict\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = 'OrderedDict'
    var_7 = bool('OrderedDict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_7 is True
    var_8 = var_1.import_index
    assert var_8 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os  # inline comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['straight']['os']
    var_5 = bool(var_1.categorized_comments['straight']['os'] == [' inline comment'])
    assert var_5 is True
    var_6 = var_1.lines_without_imports[0]
    assert var_6 == '# This is a comment'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import (join, split,\n    basename)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'join'
    var_5 = bool('join' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_5 is True
    var_6 = 'split'
    var_7 = bool('split' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_7 is True
    var_8 = 'basename'
    var_9 = bool('basename' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nimport pandas as pd\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'pandas'
    var_5 = bool('pandas' in var_1.imports['THIRDPARTY']['straight'])
    assert var_5 is True
    var_6 = var_1.as_map['straight']['numpy']
    var_7 = bool(var_1.as_map['straight']['numpy'] == ['np'])
    assert var_7 is True
    var_8 = var_1.as_map['straight']['pandas']
    var_9 = bool(var_1.as_map['straight']['pandas'] == ['pd'])
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from numpy import array as arr\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'array'
    var_5 = bool('array' in var_1.imports['THIRDPARTY']['from']['numpy'])
    assert var_5 is True
    var_6 = var_1.as_map['from']['numpy.array']
    var_7 = bool(var_1.as_map['from']['numpy.array'] == ['arr'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'my_module'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import my_module\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'my_module'
    var_8 = bool('my_module' in var_6.imports['my_module']['straight'])
    assert var_8 is True
    var_9 = 'os'
    var_10 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# STDLIB'
    var_1 = '# THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = 'section_comments'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '# STDLIB\nimport os\n# THIRDPARTY\nimport numpy\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = 'os'
    var_9 = bool('os' in var_7.imports['STDLIB']['straight'])
    assert var_9 is True
    var_10 = 'numpy'
    var_11 = bool('numpy' in var_7.imports['THIRDPARTY']['straight'])
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'numpy'
    var_5 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_5 is True
    var_6 = var_1.import_placements['# isort:imports-stdlib']
    assert var_6 == 'STDLIB'
    var_7 = var_1.import_placements['# isort:imports-thirdparty']
    assert var_7 == 'THIRDPARTY'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join, split,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('Hello')\nimport os\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = var_5.lines_without_imports[0]
    assert var_7 == "print('Hello')"

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' not in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.lines_without_imports[0]
    assert var_6 == 'import os  # isort:skip'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from numpy import array as arr  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['from']['numpy.__combined_as__']
    var_7 = bool(var_5.categorized_comments['from']['numpy.__combined_as__'] == [' comment'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os\nfrom sys import exit as exit\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' not in var_5.as_map['straight'])
    assert var_7 is True
    var_8 = 'sys.exit'
    var_9 = bool('sys.exit' not in var_5.as_map['from'])
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'else-type place_module for os returned STDLIB'
    var_7 = bool('else-type place_module for os returned STDLIB' in var_5.verbose_output)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import unknown_module\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.import_index
    assert var_4 == -1
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# Just a comment\n# Another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# Just a comment', '# Another comment'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join, \\\n    split, basename\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'join'
    var_5 = bool('join' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_5 is True
    var_6 = 'split'
    var_7 = bool('split' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_7 is True
    var_8 = 'basename'
    var_9 = bool('basename' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from numpy cimport array\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'array'
    var_5 = bool('array' in var_1.imports['THIRDPARTY']['from']['numpy'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# Above comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments['above']['straight']['os']
    var_3 = bool(var_1.categorized_comments['above']['straight']['os'] == ['# Above comment'])
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'



# Parsed testcases at query #11
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['above']['straight']['os']
    var_5 = bool(var_1.categorized_comments['above']['straight']['os'] == ['# comment'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import (join, split)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'join'
    var_5 = bool('join' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_5 is True
    var_6 = 'split'
    var_7 = bool('split' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import pandas as pd'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'pandas'
    var_3 = bool('pandas' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'pd'
    var_5 = bool('pd' in var_1.as_map['straight']['pandas'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from numpy import array as arr'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'array'
    var_5 = bool('array' in var_1.imports['THIRDPARTY']['from']['numpy'])
    assert var_5 is True
    var_6 = 'arr'
    var_7 = bool('arr' in var_1.as_map['from']['numpy.array'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_1.lines_without_imports[0]
    assert var_4 == 'import os  # isort:skip'

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.import_placements['# isort:imports-stdlib']
    assert var_4 == 'STDLIB'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import django\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'django'
    var_8 = bool('django' in var_6.imports['django']['straight'])
    assert var_8 is True
    var_9 = 'os'
    var_10 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_10 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join,'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# Just a comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_1.lines_without_imports[0]
    assert var_4 == '# Just a comment'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join, \\\n    split'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'join'
    var_5 = bool('join' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_5 is True
    var_6 = 'split'
    var_7 = bool('split' in var_1.imports['STDLIB']['from']['os.path'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from numpy cimport array'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'array'
    var_5 = bool('array' in var_1.imports['THIRDPARTY']['from']['numpy'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from numpy import array as arr  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'numpy.__combined_as__'
    var_7 = bool('numpy.__combined_as__' in var_5.categorized_comments['from'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' not in var_5.as_map['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = '# noqa'
    var_8 = bool('# noqa' not in var_6.categorized_comments['above']['straight']['os'])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# section\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import unknown_module'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join  # comment for join'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'join'
    var_3 = bool('join' in var_1.categorized_comments['nested']['os.path'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['nested']['os.path']['join']
    assert var_4 == 'comment for join'



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output
    var_9 = len(var_8)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_392_true. Retrieved 10/17 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n# comment\nimport sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.out_lines
    var_5 = len(var_4)
    var_6 = bool(var_5 > 1)
    assert var_6 is True
    var_7 = -1
    var_8 = var_4[var_7]
    var_9 = '#'
    var_10 = '"""'
    var_11 = "'''"
    var_12 = 'isort:imports-'
    var_13 = 'isort: imports-'
    var_14 = bool(not var_2.treat_all_comments_as_code)
    assert var_14 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_391_evaluates_to_false. Retrieved 39/52 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.out_lines
    var_5 = -1
    var_6 = 0
    var_7 = 'os'
    var_8 = []
    var_9 = None
    var_10 = [var_7]
    var_11 = 'import os'
    var_12 = 'straight'
    var_13 = False
    var_14 = {}
    var_15 = None
    var_16 = set()
    var_17 = 'from'
    var_18 = 'straight'
    var_19 = 'nested'
    var_20 = 'above'
    var_21 = {}
    var_22 = {}
    var_23 = {}
    var_24 = {}
    var_25 = {}
    var_26 = {var_18: var_24, var_17: var_25}
    var_27 = {var_17: var_21, var_18: var_22, var_19: var_23, var_20: var_26}
    var_28 = set()
    var_29 = lambda x: x
    var_30 = var_29(var_7)
    var_31 = []
    var_32 = 'straight'
    var_33 = 'from'
    var_34 = []
    var_35 = []
    var_36 = var_29(var_7)
    var_37 = len(var_4)
    var_38 = -1
    var_39 = 1
    var_40 = max(var_5, var_38, var_39)
    var_41 = var_40 - var_39
    var_42 = var_37 > var_41
    assert var_42 is False



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section comment'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'section_comments'
    var_4 = 'section_comments_end'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '# section comment\nimport os'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.import_index
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = []
    var_1 = '# end section'
    var_2 = [var_1]
    var_3 = 'section_comments'
    var_4 = 'section_comments_end'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '# end section\nimport os'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.import_index
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# same'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'section_comments'
    var_4 = 'section_comments_end'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '# same\nimport os'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.import_index
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section'
    var_1 = [var_0]
    var_2 = '# end'
    var_3 = [var_2]
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = 'import os'
    var_9 = module_1.file_contents(var_8, var_7)
    var_10 = var_9.import_index
    var_11 = bool(var_9.import_index != 0)
    assert var_11 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'section_comments'
    var_4 = 'section_comments_end'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '# section\n"""\nimport os'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.import_index
    var_10 = bool(var_8.import_index != 0)
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from foo import bar as baz'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'foo.__combined_as__'
    var_7 = bool('foo.__combined_as__' not in var_5.categorized_comments['from'])
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_associated_comment_not_in_comments. Retrieved 9/11 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = set()
    var_3 = None
    var_4 = 'verbose'
    var_5 = 'only_modified'
    var_6 = 'remove_redundant_aliases'
    var_7 = 'force_single_line'
    var_8 = 'treat_all_comments_as_code'
    var_9 = 'treat_comments_as_code'
    var_10 = 'line_ending'
    var_11 = {var_4: var_0, var_5: var_1, var_6: var_1, var_7: var_1, var_8: var_1, var_9: var_2, var_10: var_3}
    var_12 = module_0.Config(**var_11)
    var_13 = 'from module import something  # comment'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = var_14.verbose_output
    var_16 = 'associated_comment'



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'first_party'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import missing_module'
    var_8 = module_1.file_contents(var_7, var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_as_name_not_in_as_map_straight_module. Retrieved 5/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'straight'
    var_5 = 'module_name'
    var_6 = 'alias_name'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_true. Retrieved 14/29 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n# comment\nimport sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.out_lines
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = ''
    var_8 = '#'
    var_9 = '"""'
    var_10 = "'''"
    var_11 = 'isort:imports-'
    var_12 = 'isort: imports-'
    var_13 = var_2.treat_all_comments_as_code
    var_14 = var_2.treat_comments_as_code



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_144_evaluates_true. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import (submodule  # comment\n, another)'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'from module import (submodule  # comment\n, another as a)'
    var_6 = module_1.file_contents(var_5, var_1)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = 'from module import (submodule  # comment\n, another  # another comment\n)'
    var_9 = module_1.file_contents(var_8, var_1)
    var_10 = bool(var_9 is not None)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'print("Hello")'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('Hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = '"""'
    var_7 = (var_4, var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'''docstring"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = "'''"
    var_7 = (var_4, var_6)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring"""'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("\\"Hello\\"")'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"text" # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os; print('hi')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hi'); x = 1"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = (var_4, var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os; print('hi'); x = 1"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = (var_4, var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hi'); # comment"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = (var_4, var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hi'); x = 1"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("Hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = (var_4, var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"Hello"'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = ''
    var_8 = (var_6, var_7)
    var_9 = bool(var_5 == var_8)
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = False
    var_8 = (var_7, var_1)
    var_9 = bool(var_6 == var_8)
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = ()
    var_3 = True
    var_4 = module_0.skip_line(var_0, var_0, var_1, var_2, var_3)
    var_5 = False
    var_6 = (var_5, var_0)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# only a comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"text" \'text2\''
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = False
    var_7 = (var_6, var_1)
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_empty. Retrieved 2/3 statements.
# Partially parsed test_file_contents_only_code. Retrieved 2/3 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_1.import_index
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['from']['sys']['path']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_1.import_index
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.imports['STDLIB']['straight']['sys']
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ["print('hello')"])
    assert var_4 is True
    var_5 = var_1.import_index
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ["print('hello')"])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['# comment'])
    assert var_4 is True
    var_5 = var_1.import_index
    assert var_5 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.place_imports['STDLIB']
    var_4 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_4 is True
    var_5 = var_1.import_placements['# isort:imports-stdlib']
    assert var_5 == 'STDLIB'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.as_map['straight']['os']
    var_4 = bool(var_1.as_map['straight']['os'] == ['operating_system'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path as sys_path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['from']['sys']['path']
    assert var_2 is True
    var_3 = var_1.as_map['from']['sys.path']
    var_4 = bool(var_1.as_map['from']['sys.path'] == ['sys_path'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import (\n    path,\n    argv\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['from']['sys']['path']
    assert var_2 is True
    var_3 = var_1.imports['STDLIB']['from']['sys']['argv']
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['from']['sys']['path']
    assert var_2 is True
    var_3 = 'sys'
    var_4 = bool('sys' in var_1.trailing_commas)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path, \\\n    argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['from']['sys']['path']
    assert var_2 is True
    var_3 = var_1.imports['STDLIB']['from']['sys']['argv']
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from cython cimport boundscheck'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['from']['cython']['boundscheck']
    assert var_2 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.imports['STDLIB']['straight']['os']
    assert var_6 is True
    var_7 = var_5.lines_without_imports
    var_8 = bool(var_5.lines_without_imports == ["print('hello')"])
    assert var_8 is True
    var_9 = var_5.import_index
    assert var_9 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport tests.mock'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.imports['STDLIB']['straight']['os']
    assert var_7 is True
    var_8 = var_6.imports['tests']['straight']['tests.mock']
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'else-type place_module for os returned STDLIB'
    var_7 = bool('else-type place_module for os returned STDLIB' in var_5.verbose_output)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments['nested']['sys']['path']
    assert var_2 == '  # comment'

import isort.parse as module_0

def test_case_0():
    var_0 = '# above comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments['above']['straight']['os']
    var_3 = bool(var_1.categorized_comments['above']['straight']['os'] == ['# above comment'])
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from sys import path as sys_path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['from']['sys.__combined_as__']
    var_7 = bool(var_5.categorized_comments['from']['sys.__combined_as__'] == ['  # comment'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' not in var_5.as_map['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import unknown_module'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.lines_without_imports
    var_8 = bool(var_6.lines_without_imports == ['# noqa'])
    assert var_8 is True
    var_9 = var_6.import_index
    assert var_9 == 1



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = 'from module import item'
    var_1 = 'item'
    var_2 = [var_1]
    var_3 = -1
    var_4 = var_2[var_3]
    var_5 = ','
    var_6 = -1
    var_7 = -1
    var_8 = var_2[var_7]
    var_9 = var_0.split(var_8)[var_6]
    var_10 = var_5 in var_9
    var_11 = var_2 and var_4 and var_10
    assert var_11 is False



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'from unknown_module import something'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'could not place module unknown_module'
    var_8 = bool('could not place module unknown_module' in var_6.verbose_output[0])
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_verbose_output_contains_correct_message_for_straight_import. Retrieved 6/8 statements.
# Partially parsed test_verbose_output_contains_correct_message_for_from_import. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output
    var_9 = len(var_8)
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.verbose_output
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.verbose_output
    var_8 = 'else-type place_module for os returned'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'from os import path'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.verbose_output
    var_8 = 'else-type place_module for os returned'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output
    var_9 = len(var_8)
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'FIRSTPARTY'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from unknown_module import something'
    var_8 = module_1.file_contents(var_7, var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_397_evaluates_to_true. Retrieved 15/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'treat_all_comments_as_code'
    var_3 = 'treat_comments_as_code'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '# This is a comment'
    var_7 = '# Another comment'
    var_8 = [var_6, var_7]
    var_9 = -1
    var_10 = var_8[var_9]
    var_11 = '#'
    var_12 = '"""'
    var_13 = "'''"
    var_14 = 'isort:imports-'
    var_15 = 'isort: imports-'
    var_16 = var_5.treat_all_comments_as_code
    var_17 = var_5.treat_comments_as_code



# Parsed testcases at query #9
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['above']['straight']['os']
    var_5 = bool(var_1.categorized_comments['above']['straight']['os'] == ['# comment'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sep)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'sep'
    var_7 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = var_1.as_map['straight']['numpy']
    var_5 = bool(var_1.as_map['straight']['numpy'] == ['np'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from pandas import DataFrame as df'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'pandas'
    var_3 = bool('pandas' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'DataFrame'
    var_5 = bool('DataFrame' in var_1.imports['THIRDPARTY']['from']['pandas'])
    assert var_5 is True
    var_6 = var_1.as_map['from']['pandas.DataFrame']
    var_7 = bool(var_1.as_map['from']['pandas.DataFrame'] == ['df'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'pandas'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import pandas\nimport numpy'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'pandas'
    var_8 = bool('pandas' in var_6.imports['pandas']['straight'])
    assert var_8 is True
    var_9 = 'numpy'
    var_10 = bool('numpy' in var_6.imports['THIRDPARTY']['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# stdlib'
    var_1 = '# thirdparty'
    var_2 = [var_0, var_1]
    var_3 = 'section_comments'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '# stdlib\nimport os\n# thirdparty\nimport numpy'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = 'os'
    var_9 = bool('os' in var_7.imports['STDLIB']['straight'])
    assert var_9 is True
    var_10 = 'numpy'
    var_11 = bool('numpy' in var_7.imports['THIRDPARTY']['straight'])
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.place_imports['STDLIB'])
    assert var_3 is True
    var_4 = var_1.import_placements['# isort:imports-stdlib']
    assert var_4 == 'STDLIB'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path,)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'straight'
    var_3 = 'STDLIB'
    var_4 = var_1.imports[var_3][var_2]
    var_5 = len(var_4)
    assert var_5 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment1\n# comment2'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from pandas import DataFrame as df, Series as sr'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' in var_5.imports['THIRDPARTY']['from'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pandas'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' not in var_5.as_map['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, sep  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'path'
    var_7 = bool('path' in var_5.categorized_comments['nested']['os'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = '# noqa'
    var_8 = bool('# noqa' not in var_6.categorized_comments['above']['straight']['os'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = 'sections'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import unknown_module'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (  # comment1\n    path,  # comment2\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'path'
    var_3 = bool('path' in var_1.categorized_comments['nested']['os'])
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, \\\n    sep'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'sep'
    var_7 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc cimport math'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'libc'
    var_3 = bool('libc' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.change_count
    assert var_2 == -1



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import item1, item2  # comment'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'sys'
    var_6 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True
    var_7 = var_1.lines_without_imports
    var_8 = bool(var_1.lines_without_imports == [])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'collections'
    var_4 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'defaultdict'
    var_6 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = var_1.lines_without_imports
    var_6 = bool(var_1.lines_without_imports == ['# comment'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sep)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True
    var_7 = 'sep'
    var_8 = bool('sep' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'numpy'
    var_4 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_4 is True
    var_5 = 'np'
    var_6 = bool('np' in var_1.as_map['straight']['numpy'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'numpy'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import numpy\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'numpy'
    var_8 = bool('numpy' in var_6.imports['numpy']['straight'])
    assert var_8 is True
    var_9 = 'os'
    var_10 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# stdlib'
    var_1 = '# thirdparty'
    var_2 = [var_0, var_1]
    var_3 = 'section_comments'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '# stdlib\nimport os\n# thirdparty\nimport numpy\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.import_index
    assert var_8 == 1
    var_9 = 'os'
    var_10 = bool('os' in var_7.imports['STDLIB']['straight'])
    assert var_10 is True
    var_11 = 'numpy'
    var_12 = bool('numpy' in var_7.imports['THIRDPARTY']['straight'])
    assert var_12 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True
    var_7 = 'os'
    var_8 = bool('os' in var_1.trailing_commas)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True
    var_9 = 'place_module for os returned'
    var_10 = bool('place_module for os returned' in var_5.verbose_output[0])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = 'os'
    var_8 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_8 is True
    var_9 = var_5.lines_without_imports
    var_10 = bool(var_5.lines_without_imports == ["print('hello')"])
    assert var_10 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = 'sys'
    var_4 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = var_1.lines_without_imports
    var_6 = bool(var_1.lines_without_imports == ['import os  # isort:skip'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as operating_system\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = 'os'
    var_8 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_8 is True
    var_9 = 'operating_system'
    var_10 = bool('operating_system' in var_5.as_map['straight']['os'])
    assert var_10 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True
    var_7 = 'path'
    var_8 = bool('path' in var_1.categorized_comments['nested']['os'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# above comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = '# above comment'
    var_6 = bool('# above comment' in var_1.categorized_comments['above']['straight']['os'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment 1\n# comment 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['# comment 1', '# comment 2'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'sys'
    var_6 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'sections'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import mymodule\nimport os\n'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = 'mymodule'
    var_10 = bool('mymodule' in var_8.imports['CUSTOM']['straight'])
    assert var_10 is True
    var_11 = 'os'
    var_12 = bool('os' in var_8.imports['STDLIB']['straight'])
    assert var_12 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = 'os'
    var_8 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_8 is True
    var_9 = 'os'
    var_10 = bool('os' not in var_5.as_map['straight'])
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, sep  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = 'os'
    var_8 = bool('os' in var_5.imports['STDLIB']['from'])
    assert var_8 is True
    var_9 = 'path'
    var_10 = bool('path' in var_5.imports['STDLIB']['from']['os'])
    assert var_10 is True
    var_11 = 'sep'
    var_12 = bool('sep' in var_5.imports['STDLIB']['from']['os'])
    assert var_12 is True
    var_13 = 'path'
    var_14 = bool('path' in var_5.categorized_comments['nested']['os'])
    assert var_14 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 8/13 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/17 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 6/7 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 9/13 statements.
# Partially parsed test_file_contents_with_forced_separate. Retrieved 12/14 statements.
# Partially parsed test_file_contents_as_import. Retrieved 6/7 statements.
# Partially parsed test_file_contents_from_import_with_as. Retrieved 7/11 statements.
# Partially parsed test_file_contents_skip_comment. Retrieved 6/7 statements.
# Partially parsed test_file_contents_remove_redundant_aliases. Retrieved 4/5 statements.
# Partially parsed test_file_contents_empty_file. Retrieved 2/3 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 8/9 statements.
# Partially parsed test_file_contents_backslash_continuation. Retrieved 7/11 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = 'os'
    var_5 = True
    var_6 = (var_4, var_5)
    var_7 = 'sys'
    var_8 = (var_7, var_5)
    var_9 = [var_6, var_8]
    var_10 = [var_9]
    var_11 = var_1.imports['STDLIB']['straight']
    var_12 = []
    var_13 = var_1.imports['THIRDPARTY']['straight']
    var_14 = []
    var_15 = var_1.imports['FIRSTPARTY']['straight']
    var_16 = []
    var_17 = var_1.imports['LOCALFOLDER']['straight']
    var_18 = var_1.import_index
    assert var_18 == 0
    var_19 = var_1.change_count
    assert var_19 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict\nfrom os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = 'defaultdict'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = 'os'
    var_9 = 'path'
    var_10 = (var_9, var_4)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = var_1.imports['STDLIB']['from']
    var_14 = var_1.import_index
    assert var_14 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os  # inline comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.categorized_comments['straight']['os']
    var_9 = bool(var_1.categorized_comments['straight']['os'] == ['# inline comment'])
    assert var_9 is True
    var_10 = var_1.lines_without_imports
    var_11 = bool(var_1.lines_without_imports == ['# comment'])
    assert var_11 is True
    var_12 = var_1.import_index
    assert var_12 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sep)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = 'sep'
    var_7 = (var_6, var_4)
    var_8 = [var_5, var_7]
    var_9 = [var_8]
    var_10 = var_1.imports['STDLIB']['from']

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'forced'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import forced_module\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'forced_module'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = var_6.imports['forced']['straight']
    var_13 = 'os'
    var_14 = (var_13, var_8)
    var_15 = [var_14]
    var_16 = [var_15]
    var_17 = var_6.imports['STDLIB']['straight']

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.as_map['straight']['os']
    var_9 = bool(var_1.as_map['straight']['os'] == ['operating_system'])
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']
    var_9 = var_1.as_map['from']['os.path']
    var_10 = bool(var_1.as_map['from']['os.path'] == ['p'])
    assert var_10 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.lines_without_imports
    var_9 = bool(var_1.lines_without_imports == ['import os  # isort:skip'])
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_placements['# isort:imports-stdlib']
    assert var_2 == 'STDLIB'
    var_3 = var_1.place_imports['STDLIB']
    var_4 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas
    var_3 = bool(var_1.trailing_commas == {'os'})
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = var_5.lines_without_imports
    var_8 = bool(var_5.lines_without_imports == ["print('hello')"])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'else-type place_module for os returned STDLIB'
    var_7 = bool('else-type place_module for os returned STDLIB' in var_5.verbose_output)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'CUSTOM'
    var_1 = [var_0]
    var_2 = 'sections'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path as p  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['from']['os.__combined_as__']
    var_7 = bool(var_5.categorized_comments['from']['os.__combined_as__'] == ['# comment'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.as_map['straight']

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['nested']['os']['path']
    assert var_6 == '# comment'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# special'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# special\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.lines_without_imports
    var_8 = bool(var_6.lines_without_imports == ['# special'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = var_1.import_index
    assert var_4 == -1
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.verbose_output
    var_8 = bool(var_6.verbose_output == [])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = 'sys'
    var_6 = (var_5, var_3)
    var_7 = [var_4, var_6]
    var_8 = [var_7]
    var_9 = var_1.imports['STDLIB']['straight']

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']



# Parsed testcases at query #13
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'some line without quotes'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5[0]
    assert var_6 is False



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'from a import b, c,'
    var_1 = []
    var_2 = set()
    var_3 = 'a'
    var_4 = -1
    var_5 = var_1[var_4]
    var_6 = ','
    var_7 = -1
    var_8 = -1
    var_9 = var_1[var_8]
    var_10 = var_0.split(var_9)[var_7]
    var_11 = var_6 in var_10
    var_12 = var_1 and var_5 and var_11
    var_13 = bool(not var_12)
    assert var_13 is True
    var_14 = bool(var_3 not in var_2)
    assert var_14 is True

def test_case_0():
    var_0 = 'from a import b, c,'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = ''
    var_4 = [var_1, var_2, var_3]
    var_5 = set()
    var_6 = 'a'
    var_7 = -1
    var_8 = var_4[var_7]
    var_9 = ','
    var_10 = -1
    var_11 = -1
    var_12 = var_4[var_11]
    var_13 = var_0.split(var_12)[var_10]
    var_14 = var_9 in var_13
    var_15 = var_4 and var_8 and var_14
    var_16 = bool(not var_15)
    assert var_16 is True
    var_17 = bool(var_6 not in var_5)
    assert var_17 is True

def test_case_0():
    var_0 = 'from a import b, c'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_1, var_2]
    var_4 = set()
    var_5 = 'a'
    var_6 = -1
    var_7 = var_3[var_6]
    var_8 = ','
    var_9 = -1
    var_10 = -1
    var_11 = var_3[var_10]
    var_12 = var_0.split(var_11)[var_9]
    var_13 = var_8 in var_12
    var_14 = var_3 and var_7 and var_13
    var_15 = bool(not var_14)
    assert var_15 is True
    var_16 = bool(var_5 not in var_4)
    assert var_16 is True

def test_case_0():
    var_0 = 'from a import b'
    var_1 = 'b'
    var_2 = [var_1]
    var_3 = set()
    var_4 = 'a'
    var_5 = -1
    var_6 = var_2[var_5]
    var_7 = ','
    var_8 = -1
    var_9 = -1
    var_10 = var_2[var_9]
    var_11 = var_0.split(var_10)[var_8]
    var_12 = var_7 in var_11
    var_13 = var_2 and var_6 and var_12
    var_14 = bool(not var_13)
    assert var_14 is True
    var_15 = bool(var_4 not in var_3)
    assert var_15 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 13/27 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# This is a comment'
    var_3 = 'import os'
    var_4 = [var_2, var_3]
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = '#'
    var_8 = '"""'
    var_9 = "'''"
    var_10 = 'isort:imports-'
    var_11 = 'isort: imports-'
    var_12 = var_1.treat_all_comments_as_code
    var_13 = var_1.treat_comments_as_code



