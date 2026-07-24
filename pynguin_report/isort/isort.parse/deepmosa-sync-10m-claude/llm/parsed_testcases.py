####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.change_count
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_6 is True
    var_7 = var_3.change_count
    assert var_7 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_6 is True
    var_7 = 'path'
    var_8 = bool('path' in var_3.imports['STDLIB']['from']['os'])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_6 is True
    var_7 = 'sys'
    var_8 = bool('sys' in var_3.imports['STDLIB']['straight'])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "x = 1\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_3.import_index
    assert var_6 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n\ndef foo():\n    pass\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # comment\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    sep\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os as operating_system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.as_map['straight'])
    assert var_6 is True
    var_7 = 'operating_system'
    var_8 = bool('operating_system' in var_3.as_map['straight']['os'])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path as p\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os.path'
    var_6 = bool('os.path' in var_3.as_map['from'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = len(var_4)
    var_6 = bool(var_5 >= 1)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = 'sys'
    var_7 = bool('sys' in var_3.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import \\\n    path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: split\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort: skip\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    var_7 = var_4 - var_6
    var_8 = var_3.change_count
    var_9 = bool(var_3.change_count == var_7)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path  # comment\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from libc.stdlib cimport malloc\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,  # path comment\n    sep\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path,\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.trailing_commas)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 6/7 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.imports
    var_10 = bool(var_3.imports == {section: {'straight': {}, 'from': {}} for section in var_1.sections + var_1.forced_separate})
    assert var_10 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "print('hello')\nprint('world')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = 'sys'
    var_7 = bool('sys' in var_3.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import numpy as np\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'numpy'
    var_5 = bool('numpy' in var_3.as_map['straight'])
    assert var_5 is True
    var_6 = 'np'
    var_7 = bool('np' in var_3.as_map['straight']['numpy'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path as p\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os.path'
    var_5 = bool('os.path' in var_3.as_map['from'])
    assert var_5 is True
    var_6 = 'p'
    var_7 = bool('p' in var_3.as_map['from']['os.path'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    environ\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_5 is True
    var_6 = 'path'
    var_7 = bool('path' in var_3.imports['STDLIB']['from']['os'])
    assert var_7 is True
    var_8 = 'environ'
    var_9 = bool('environ' in var_3.imports['STDLIB']['from']['os'])
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = 'os'
    var_7 = bool('os' in var_3.categorized_comments['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines[-1]
    assert var_4 == ''

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = 'sys'
    var_7 = bool('sys' in var_3.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import \\\n    path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_5 is True
    var_6 = 'path'
    var_7 = bool('path' in var_3.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: split\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: skip\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    var_7 = var_4 - var_6
    var_8 = var_3.change_count
    var_9 = bool(var_3.change_count == var_7)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nimport sys\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    assert var_4 == 3

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, environ\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['from'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from libc.stdlib cimport malloc\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\nfrom os import environ\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_5 is True
    var_6 = 'path'
    var_7 = bool('path' in var_3.imports['STDLIB']['from']['os'])
    assert var_7 is True
    var_8 = 'environ'
    var_9 = bool('environ' in var_3.imports['STDLIB']['from']['os'])
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.trailing_commas)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n\ndef foo():\n    pass\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['straight'])
    assert var_5 is True
    var_6 = 'def foo():'
    var_7 = bool('def foo():' in var_3.lines_without_imports)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path  # comment\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.categorized_comments['from'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,  # path module\n    environ,  # environment\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports['STDLIB']['from'])
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_391_evaluates_to_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 391 evaluates to False when out_lines is empty.'
    var_1 = []
    var_2 = -1
    var_3 = var_1[var_2]
    var_4 = ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 1/3 statements.
# Partially parsed test_file_contents_no_imports. Retrieved 1/5 statements.
# Partially parsed test_file_contents_single_import. Retrieved 5/10 statements.
# Partially parsed test_file_contents_from_import. Retrieved 5/10 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 5/10 statements.
# Partially parsed test_file_contents_import_with_comment. Retrieved 5/10 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 5/10 statements.
# Partially parsed test_file_contents_import_with_alias. Retrieved 1/3 statements.
# Partially parsed test_file_contents_from_import_with_alias. Retrieved 1/3 statements.
# Partially parsed test_file_contents_mixed_imports_and_code. Retrieved 1/5 statements.
# Partially parsed test_file_contents_import_with_trailing_comma. Retrieved 1/3 statements.
# Partially parsed test_file_contents_section_comment. Retrieved 1/3 statements.
# Partially parsed test_file_contents_skip_line_with_quote. Retrieved 1/3 statements.
# Partially parsed test_file_contents_line_separator_inference. Retrieved 1/3 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 1/5 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 1/3 statements.
# Partially parsed test_file_contents_semicolon_separated_statements. Retrieved 5/10 statements.
# Partially parsed test_file_contents_escaped_line. Retrieved 1/3 statements.
# Partially parsed test_file_contents_cimport. Retrieved 1/3 statements.
# Partially parsed test_file_contents_change_count. Retrieved 1/7 statements.
# Partially parsed test_file_contents_sections_initialized. Retrieved 1/4 statements.
# Partially parsed test_file_contents_as_map_structure. Retrieved 3/9 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'straight'
    var_4 = {}
    var_5 = 'os'

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'from'
    var_4 = {}
    var_5 = 'os'

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'straight'
    var_4 = {}
    var_5 = 'os'
    var_6 = 'sys'

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'straight'
    var_4 = {}
    var_5 = 'os'

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'from'
    var_4 = {}
    var_5 = 'os'

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = 'operating_system'

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = 'p'

def test_case_0():
    var_0 = 'import os\n\nx = 1\nimport sys\n'

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = 'os'

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport numpy\n'
    var_1 = 'THIRDPARTY'

def test_case_0():
    var_0 = '"""\nDocstring\n"""\nimport os\n'

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'

def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = 'from os import path  # path comment\n'

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = 0
    var_2 = {}
    var_3 = 'straight'
    var_4 = {}
    var_5 = 'os'
    var_6 = 'sys'

def test_case_0():
    var_0 = 'from os import \\\n    path\n'

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'

def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'straight'
    var_2 = 'from'

def test_case_0():
    var_0 = 'import os as o\nfrom sys import path as p\n'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = 'straight'
    var_4 = 'from'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_skip_line. Retrieved 6/9 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_6 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['from'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'environ'
    var_7 = bool('environ' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.as_map['straight'])
    assert var_3 is True
    var_4 = 'operating_system'
    var_5 = bool('operating_system' in var_1.as_map['straight']['os'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.as_map['from'])
    assert var_3 is True
    var_4 = 'p'
    var_5 = bool('p' in var_1.as_map['from']['os.path'])
    assert var_5 is True

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
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '\n'
    var_2 = 'line_ending'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.line_separator
    assert var_6 == '\n'

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
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: split\nimport os\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os  # isort:skip\nprint('test')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = 'STDLIB'
    var_3 = {}
    var_4 = 'straight'
    var_5 = {}
    var_6 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = '"""Module docstring"""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'treat_all_comments_as_code'
    var_3 = 'treat_comments_as_code'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '# This is a comment\nimport os\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = str(var_7)
    var_10 = 'os'
    var_11 = bool('os' in var_9)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'treat_all_comments_as_code'
    var_3 = 'treat_comments_as_code'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = '# This is a regular comment\nimport os\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.lines_without_imports
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_place_imports_directive. Retrieved 2/3 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.lines_without_imports[0]
    assert var_5 == 'x = 1'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines[-1]
    assert var_2 == ''
    var_3 = var_1.original_line_count
    assert var_3 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'environ'
    var_7 = bool('environ' in var_1.imports['STDLIB']['from']['os'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.as_map['straight'])
    assert var_3 is True
    var_4 = 'operating_system'
    var_5 = bool('operating_system' in var_1.as_map['straight']['os'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = bool('os.path' in var_1.as_map['from'])
    assert var_3 is True
    var_4 = 'p'
    var_5 = bool('p' in var_1.as_map['from']['os.path'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: skip\nimport b\nimport a\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 >= 1)
    assert var_4 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, environ\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['from'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

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
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'x = 1'
    var_4 = bool('x = 1' in var_1.lines_without_imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'THIRDPARTY'
    var_3 = bool('THIRDPARTY' in var_1.place_imports)
    assert var_3 is True
    var_4 = 'THIRDPARTY'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv\nimport json\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['from'])
    assert var_5 is True
    var_6 = 'json'
    var_7 = bool('json' in var_1.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdio cimport printf\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 5/6 statements.
# Partially parsed test_file_contents_nested_comment_from_import. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.change_count
    assert var_9 == 0
    var_10 = var_3.original_line_count
    assert var_10 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'os'
    var_6 = bool('os' in var_3.imports)
    assert var_6 is True
    var_7 = var_3.change_count
    assert var_7 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.change_count
    assert var_5 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.change_count
    assert var_5 == -2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import numpy as np\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'np'
    var_6 = bool('np' in var_3.as_map['straight']['numpy'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path as p\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'p'
    var_6 = bool('p' in var_3.as_map['from']['os.path'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    environ\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.change_count
    assert var_5 == -4

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'straight'
    var_6 = var_3.categorized_comments[var_5]
    var_7 = len(var_6)
    var_8 = bool(var_7 >= 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "print('hello')\nimport os\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 1
    var_5 = var_3.lines_without_imports[0]
    assert var_5 == "print('hello')"

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = "print('hello')"
    var_6 = bool("print('hello')" in var_3.lines_without_imports)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: split\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort: skip\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os, \\\n    sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nprint('test')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    assert var_4 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True
    var_7 = var_3.sections
    var_8 = bool(var_3.sections == var_1.sections)
    assert var_8 is True

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.trailing_commas)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path  # comment\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.categorized_comments

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:imports-FUTURE\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'FUTURE'
    var_5 = bool('FUTURE' in var_3.place_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: split\nimport os\n# isort: split\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from libc.stdlib cimport malloc\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, environ\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path as p\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import \\\n    (path,\n    environ)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# This is a comment\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_file_contents_verbose_mode. Retrieved 5/8 statements.
# Partially parsed test_file_contents_place_imports. Retrieved 3/6 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.in_lines
    var_6 = bool(var_1.in_lines == [])
    assert var_6 is True
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines
    var_4 = bool(var_1.in_lines == ['import os', ''])
    assert var_4 is True
    var_5 = var_1.imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines
    var_4 = bool(var_1.in_lines == ['from os import path', ''])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines
    var_4 = len(var_3)
    assert var_4 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 4)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'straight'
    var_4 = var_1.as_map[var_3]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nprint(x)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines[-1]
    assert var_2 == ''
    var_3 = var_1.line_separator
    assert var_3 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 80
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: stdlib'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# isort: stdlib\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    var_8 = bool(var_6.import_index >= 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = var_2 - var_4
    var_6 = var_1.change_count
    var_7 = bool(var_1.change_count == var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, sep\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.trailing_commas
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,  # path module\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True
    var_4 = var_1.place_imports

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1\nimport os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 199 evaluates to True for import statements.'
    var_1 = 'from os import path'
    var_2 = module_0.file_contents(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = 'from libc.stdlib cimport malloc'
    var_5 = module_0.file_contents(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'import os'
    var_8 = module_0.file_contents(var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True
    var_10 = 'cimport numpy'
    var_11 = module_0.file_contents(var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True
    var_13 = 'from os import (\n    path,\n    sep\n)'
    var_14 = module_0.file_contents(var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# Custom Section'
    var_1 = [var_0]
    var_2 = '# End Section'
    var_3 = [var_2]
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '# Custom Section\nimport os\n'
    var_9 = module_1.file_contents(var_8, var_7)
    var_10 = bool(var_9 is not None)
    assert var_10 is True
    var_11 = []
    var_12 = [var_2]
    var_13 = 'section_comments'
    var_14 = 'section_comments_end'
    var_15 = {var_13: var_11, var_14: var_12}
    var_16 = module_0.Config(**var_15)
    var_17 = '# End Section\nimport os\n'
    var_18 = module_1.file_contents(var_17, var_16)
    var_19 = bool(var_18 is not None)
    assert var_19 is True
    var_20 = '# STDLIB'
    var_21 = [var_20]
    var_22 = []
    var_23 = 'section_comments'
    var_24 = 'section_comments_end'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = '# STDLIB\nimport sys\n'
    var_28 = module_1.file_contents(var_27, var_26)
    var_29 = bool(var_28 is not None)
    assert var_29 is True



# Parsed testcases at query #13
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 199 evaluates to True for import statements.'
    var_1 = 'from os import path'
    var_2 = module_0.file_contents(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = 'from libc.stdlib cimport malloc'
    var_5 = module_0.file_contents(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'import os'
    var_8 = module_0.file_contents(var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True
    var_10 = 'cimport numpy'
    var_11 = module_0.file_contents(var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True
    var_13 = 'from os import (\n    path\n)'
    var_14 = module_0.file_contents(var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True
    var_16 = 'from os import \\\n    path'
    var_17 = module_0.file_contents(var_16)
    var_18 = bool(var_17 is not None)
    assert var_18 is True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import nested_module as alias_name\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = module_1.file_contents(var_2, var_1)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 8/11 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 5/6 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.change_count
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['import os', ''])
    assert var_5 is True
    var_6 = var_3.import_index
    var_7 = bool(var_3.import_index >= 0)
    assert var_7 is True
    var_8 = 'STDLIB'
    var_9 = {}
    var_10 = 'straight'
    var_11 = {}
    var_12 = {}
    var_13 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['from os import path', ''])
    assert var_5 is True
    var_6 = var_3.import_index
    var_7 = bool(var_3.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['import os', 'import sys', ''])
    assert var_5 is True
    var_6 = var_3.import_index
    var_7 = bool(var_3.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\n\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True
    var_7 = 'print'
    var_8 = bool('print' in var_3.lines_without_imports[-1])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    environ\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import numpy as np\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.categorized_comments
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['import os'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['import os', ''])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\r'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == ['import os', ''])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os  # isort: skip\nprint('hello')\n"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'import os  # isort: skip'
    var_5 = bool('import os  # isort: skip' in var_3.lines_without_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# Custom Section'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# Custom Section\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    var_8 = bool(var_6.import_index >= 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:imports-THIRDPARTY\nimport numpy\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'THIRDPARTY'
    var_5 = bool('THIRDPARTY' in var_3.place_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os, \\\n    sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path,\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.trailing_commas
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'in_lines'
    var_5 = hasattr(var_3, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'lines_without_imports'
    var_8 = hasattr(var_3, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'import_index'
    var_11 = hasattr(var_3, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'imports'
    var_14 = hasattr(var_3, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'categorized_comments'
    var_17 = hasattr(var_3, var_16)
    var_18 = bool(var_17)
    assert var_18 is True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 308 evaluates to False when placed_module is empty string.'
    var_1 = 'mymodule'
    var_2 = [var_1]
    var_3 = 'known_first_party'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'from mymodule import something'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_file_contents_return_type. Retrieved 2/5 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = str(var_2)
    var_4 = 'os'
    var_5 = bool('os' in var_3)
    assert var_5 is True
    var_6 = var_1.import_index
    var_7 = bool(var_1.import_index >= 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = str(var_2)
    var_4 = 'os'
    var_5 = bool('os' in var_3)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True
    var_4 = var_1.imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\n\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = str(var_2)
    var_4 = 'os'
    var_5 = bool('os' in var_3)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'straight'
    var_3 = var_1.as_map[var_2]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5
    var_7 = 'numpy'
    var_8 = var_1.imports
    var_9 = str(var_8)
    var_10 = var_7 in var_9
    var_11 = bool(var_6 or var_10)
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from'
    var_3 = var_1.as_map[var_2]
    var_4 = len(var_3)
    var_5 = 0
    var_6 = var_4 > var_5
    var_7 = 'os'
    var_8 = var_1.imports
    var_9 = str(var_8)
    var_10 = var_7 in var_9
    var_11 = bool(var_6 or var_10)
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= -1)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "# header comment\nimport os\nprint('test')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas
    var_3 = len(var_2)
    var_4 = bool(var_3 >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os, \\\n    sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # inline comment\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os  # isort:skip\nprint('test')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= -1)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nimport sys\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= -1)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from . import module\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_line_335_predicate_evaluates_to_true. Retrieved 7/10 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 335 (comments and attach_comments_to is None) evaluates to True.'
    var_1 = 'from module import name  # comment\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.file_contents(var_1, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.in_lines
    var_7 = var_4.in_lines
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_6 is True
    var_7 = 'path'
    var_8 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'straight'
    var_6 = var_1.categorized_comments[var_5]
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)\n'
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
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.as_map['straight'])
    assert var_4 is True
    var_5 = 'operating_system'
    var_6 = bool('operating_system' in var_1.as_map['straight']['os'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os.path'
    var_4 = bool('os.path' in var_1.as_map['from'])
    assert var_4 is True
    var_5 = 'p'
    var_6 = bool('p' in var_1.as_map['from']['os.path'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = 'x = 1'
    var_4 = bool('x = 1' in var_1.lines_without_imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.original_line_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2
    var_3 = var_1.in_lines[-1]
    assert var_3 == ''

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
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 100
    var_1 = 'line_length'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = 'os'
    var_8 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.change_count
    assert var_2 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section comment'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# section comment\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.lines_without_imports[0])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1\nimport os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nfrom os import sep\n'
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
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

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

import isort.parse as module_0

def test_case_0():
    var_0 = '"""\nModule docstring\n"""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 3
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.
# Partially parsed test_file_contents_trailing_commas. Retrieved 3/6 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.imports
    var_12 = bool(var_1.imports == {})
    assert var_12 is True
    var_13 = var_1.categorized_comments
    var_14 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.change_count
    assert var_15 == 0
    var_16 = var_1.original_line_count
    assert var_16 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', ''])
    assert var_3 is True
    var_4 = var_1.import_index
    var_5 = bool(var_1.import_index >= 0)
    assert var_5 is True
    var_6 = var_1.original_line_count
    assert var_6 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path', ''])
    assert var_3 is True
    var_4 = var_1.import_index
    var_5 = bool(var_1.import_index >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'import sys', ''])
    assert var_3 is True
    var_4 = var_1.original_line_count
    assert var_4 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os  # comment', ''])
    assert var_3 is True
    var_4 = var_1.import_index
    var_5 = bool(var_1.import_index >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = var_1.import_index
    var_5 = bool(var_1.import_index >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\n\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = "print('hello')"
    var_3 = bool("print('hello')" in var_1.lines_without_imports)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.original_line_count
    assert var_4 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True
    var_4 = 'straight'
    var_5 = var_1.as_map[var_4]
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'import os  # isort:skip'
    var_3 = bool('import os  # isort:skip' in var_1.lines_without_imports)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: split\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, sep\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 'hello"
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, "'"))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, '"'))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = """hello'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, '"""'))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = '''hello"
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, "'''"))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'continuing string'
    var_1 = "'"
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, "'"))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "end'"
    var_1 = "'"
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'end"'
    var_1 = '"'
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'end"""'
    var_1 = '"""'
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello\\"'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (True, '"'))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "test" # comment'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (True, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from sys import path'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (False, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy; import os'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (False, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = False
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (False, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; # x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (False, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "a" + "b"'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "test#hash"'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (True, ''))
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os;;'
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = True
    var_6 = module_0.skip_line(var_0, var_1, var_2, var_4, var_5)
    var_7 = bool(var_6 == (False, ''))
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_with_skip_comment. Retrieved 6/9 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_6 is True
    var_7 = 'path'
    var_8 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'straight'
    var_6 = var_1.categorized_comments[var_5]
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True
    var_7 = 'environ'
    var_8 = bool('environ' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'operating_system'
    var_6 = bool('operating_system' in var_1.as_map['straight']['os'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = 'sys'
    var_6 = bool('sys' in var_1.imports['STDLIB']['from'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\n\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.trailing_commas)
    assert var_4 is True

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
    var_0 = '\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort: skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'straight'
    var_6 = {}
    var_7 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True

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
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

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

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nimport sys\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os.path'
    var_4 = bool('os.path' in var_1.as_map['from'])
    assert var_4 is True
    var_5 = 'p'
    var_6 = bool('p' in var_1.as_map['from']['os.path'])
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: section'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# isort: section\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    var_8 = bool(var_6.import_index >= 0)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_file_contents_from_import. Retrieved 3/7 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 3/7 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 6/9 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0
    var_6 = var_1.original_line_count
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nx = 1"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == -2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.original_line_count
    assert var_3 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    getcwd\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'numpy'
    var_4 = bool('numpy' in var_1.as_map['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n# isort: skip\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.sections
    var_5 = bool(var_3.sections == var_1.sections)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n\ndef main():\n    pass'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 3)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '"""Module docstring"""\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = "'''\nMultiline\nstring\n'''\nimport os"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 4

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, getcwd'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.trailing_commas
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_sections_initialized. Retrieved 5/7 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.in_lines
    var_6 = bool(var_1.in_lines == [])
    assert var_6 is True
    var_7 = var_1.change_count
    assert var_7 == 0
    var_8 = var_1.original_line_count
    assert var_8 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 1)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2
    var_3 = var_1.in_lines[-1]
    assert var_3 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 1)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = '"""Module docstring."""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, environ\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from . import module\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# important comment'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# important comment\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = '\r\n'
    var_2 = 'line_ending'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 270 evaluates to False.'
    var_1 = 'import os as operating_system\nimport os as operating_system\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.file_contents(var_1, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_340_evaluates_to_true. Retrieved 4/7 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 340 evaluates to True.'
    var_1 = '# This is a comment\nfrom module import something\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.file_contents(var_1, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 352 (if out_lines:) evaluates to True.'
    var_1 = '# Comment above import\nfrom module import something\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.file_contents(var_1, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4.import_index
    var_7 = len(var_6)
    var_8 = bool(var_7 >= 0)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_file_contents_returns_parsed_content. Retrieved 2/4 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.change_count
    assert var_5 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = str(var_3)
    var_5 = 'os'
    var_6 = bool('os' in var_4)
    assert var_6 is True
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.as_map['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'from'
    var_4 = var_1.as_map[var_3]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2
    var_3 = var_1.in_lines[-1]
    assert var_3 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.change_count
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""Module docstring"""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = '# Comment\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_placements
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = True
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_1, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.file_contents(var_0, var_5)
    var_7 = var_6.verbose_output

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALIMPORT'
    var_5 = 'CUSTOM'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'sections'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(**var_8)
    var_10 = 'from unknown_module import something\n'
    var_11 = module_1.file_contents(var_10, var_9)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_section_comments_predicate_evaluates_to_true. Retrieved 12/16 statements.
# Partially parsed test_section_comments_end_predicate_evaluates_to_true. Retrieved 12/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '# Section: Custom'
    var_1 = [var_0]
    var_2 = '# End Section'
    var_3 = [var_2]
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '# Section: Custom'
    var_9 = False
    var_10 = var_7.section_comments
    var_11 = var_8 in var_10
    var_12 = var_7.section_comments_end
    var_13 = var_8 in var_12
    var_14 = var_11 or var_13

import isort.settings as module_0

def test_case_0():
    var_0 = '# Section: Custom'
    var_1 = [var_0]
    var_2 = '# End Section'
    var_3 = [var_2]
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '# End Section'
    var_9 = False
    var_10 = var_7.section_comments
    var_11 = var_8 in var_10
    var_12 = var_7.section_comments_end
    var_13 = var_8 in var_12
    var_14 = var_11 or var_13



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_line_371_predicate_evaluates_to_true. Retrieved 3/6 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something  # comment1\n# comment2\nfrom module import other\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 6/10 statements.
# Partially parsed test_file_contents_from_import. Retrieved 6/10 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 8/13 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 6/9 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 6/9 statements.
# Partially parsed test_file_contents_import_with_alias. Retrieved 6/8 statements.
# Partially parsed test_file_contents_from_import_with_alias. Retrieved 6/8 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 8/13 statements.
# Partially parsed test_file_contents_backslash_continuation. Retrieved 6/9 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'straight'
    var_6 = {}
    var_7 = 'os'
    var_8 = var_1.change_count
    assert var_8 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'from'
    var_6 = {}
    var_7 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'straight'
    var_6 = {}
    var_7 = 'os'
    var_8 = {}
    var_9 = {}
    var_10 = 'sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'straight'
    var_6 = {}
    var_7 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = {}
    var_5 = 'from'
    var_6 = {}
    var_7 = 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'straight'
    var_4 = var_1.as_map[var_3]
    var_5 = 'numpy'
    var_6 = []
    var_7 = 'np'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'from'
    var_4 = var_1.as_map[var_3]
    var_5 = 'os.path'
    var_6 = []
    var_7 = 'p'

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0

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
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'STDLIB'
    var_3 = {}
    var_4 = 'straight'
    var_5 = {}
    var_6 = 'os'
    var_7 = {}
    var_8 = {}
    var_9 = 'sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'STDLIB'
    var_3 = {}
    var_4 = 'from'
    var_5 = {}
    var_6 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = '\n'
    var_2 = 'line_ending'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.line_separator
    assert var_6 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'x = 1'
    var_3 = bool('x = 1' in var_1.lines_without_imports)
    assert var_3 is True
    var_4 = 'y = 2'
    var_5 = bool('y = 2' in var_1.lines_without_imports)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_associated_comment_predicate_at_line_259. Retrieved 3/5 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import submodule as alias  # comment for alias\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # NOQA'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort: skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort: split'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = '# just a comment'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, sep'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport cython'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = '  import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa  \n'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 339 evaluates to False.'
    var_1 = []
    var_2 = -1
    var_3 = len(var_1)
    var_4 = 1
    var_5 = max(var_2, var_4)
    var_6 = var_5 - var_4
    var_7 = var_3 > var_6
    assert var_7 is False
    var_8 = []
    var_9 = -1
    var_10 = len(var_8)
    var_11 = max(var_9, var_4)
    var_12 = var_11 - var_4
    var_13 = var_10 > var_12
    assert var_13 is False
    var_14 = []
    var_15 = 0
    var_16 = len(var_14)
    var_17 = max(var_15, var_4)
    var_18 = var_17 - var_4
    var_19 = var_16 > var_18
    assert var_19 is False
    var_20 = []
    var_21 = 1
    var_22 = len(var_20)
    var_23 = max(var_21, var_4)
    var_24 = var_23 - var_4
    var_25 = var_22 > var_24
    assert var_25 is False



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 274 evaluates to False.'
    var_1 = 'import os\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1.file_contents(var_1, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_320_evaluates_to_true. Retrieved 4/7 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import name  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_file_contents_returns_parsed_content. Retrieved 2/4 statements.
# Partially parsed test_file_contents_categorized_comments. Retrieved 3/5 statements.
# Partially parsed test_file_contents_as_map. Retrieved 3/5 statements.
# Partially parsed test_file_contents_imports_structure. Retrieved 3/5 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 5/7 statements.
# Partially parsed test_file_contents_place_imports. Retrieved 3/5 statements.
# Partially parsed test_file_contents_trailing_commas. Retrieved 3/5 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = var_1.change_count
    var_7 = bool(var_1.change_count >= 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = var_1.change_count
    var_6 = bool(var_1.change_count >= 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nx = 1'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path, \\\n    sep'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    var_3 = bool(var_1.line_separator is not None)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment\nx = 1\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments
    var_3 = 'from'
    var_4 = bool('from' in var_1.categorized_comments)
    assert var_4 is True
    var_5 = 'straight'
    var_6 = bool('straight' in var_1.categorized_comments)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.as_map
    var_3 = 'straight'
    var_4 = bool('straight' in var_1.as_map)
    assert var_4 is True
    var_5 = 'from'
    var_6 = bool('from' in var_1.as_map)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = var_1.change_count
    var_4 = bool(var_1.change_count >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os'
    var_1 = True
    var_2 = 'verbose'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.verbose_output

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-FUTURE\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_66_evaluates_to_true. Retrieved 6/10 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1\nimport os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = '__getitem__'
    var_8 = hasattr(var_5, var_7)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_file_contents_verbose_mode. Retrieved 5/8 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0
    var_6 = var_1.original_line_count
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nprint('world')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.change_count
    assert var_5 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = str(var_3)
    var_5 = 'os'
    var_6 = bool('os' in var_4)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'numpy'
    var_4 = bool('numpy' in var_1.as_map['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-STDLIB\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'STDLIB'
    var_3 = bool('STDLIB' in var_1.place_imports)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    var_3 = bool(var_1.line_separator in ('\r\n', '\n'))
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path, environ\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\n\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from . import module\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path as p,  # path alias\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'STDLIB'
    var_4 = bool('STDLIB' in var_1.imports)
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_6 is True
    var_7 = 'path'
    var_8 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # system module\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True
    var_7 = 'environ'
    var_8 = bool('environ' in var_1.imports['STDLIB']['from']['os'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'numpy'
    var_4 = bool('numpy' in var_1.as_map['straight'])
    assert var_4 is True
    var_5 = 'np'
    var_6 = bool('np' in var_1.as_map['straight']['numpy'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os.path'
    var_4 = bool('os.path' in var_1.as_map['from'])
    assert var_4 is True
    var_5 = 'p'
    var_6 = bool('p' in var_1.as_map['from']['os.path'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.original_line_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_1.lines_without_imports[0]
    assert var_4 == 'x = 1'
    var_5 = var_1.lines_without_imports[1]
    assert var_5 == 'y = 2'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: split'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# isort: split\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1\nimport os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_4 is True
    var_5 = 'path'
    var_6 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    var_3 = bool(var_1.line_separator in ('\n', '\r\n', '\r'))
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,  # path comment\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.categorized_comments['nested'])
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_1 = module_0.file_contents(var_0)



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'from os import path\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = 'import_index'
    var_9 = hasattr(var_6, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 6/7 statements.
# Partially parsed test_file_contents_trailing_comma_detection. Retrieved 4/5 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 4/5 statements.
# Partially parsed test_file_contents_place_imports_marker. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.change_count
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "print('hello')\nx = 1"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_3.import_index
    assert var_6 == 0
    var_7 = 'print'
    var_8 = bool('print' in var_3.lines_without_imports[0])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports)
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "from os import path\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.imports)
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\nimport sys\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.change_count
    var_6 = bool(var_3.change_count >= 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines[-1]
    assert var_4 == ''
    var_5 = var_3.original_line_count
    assert var_5 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "from os import (\n    path,\n    sep\n)\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import numpy as np\nprint(np)'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'numpy'
    var_5 = bool('numpy' in var_3.as_map['straight'])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os  # operating system\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.categorized_comments
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "# isort: split\nimport os\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os  # isort: skip\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'os'
    var_5 = bool('os' in var_3.lines_without_imports[0])
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "from libc.stdlib cimport malloc\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= -1)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os; import sys\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os, \\\n    sys\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = "import os\nprint('hello')"
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "from os import path, sep\nprint('hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""Module docstring."""\nimport os\nprint("hello")'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "from os import (\n    path,\n)\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.trailing_commas

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "from os import path  # path comment\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.categorized_comments

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "x = 1\nimport os\nprint('hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    var_5 = bool(var_3.line_separator in ('\n', '\r\n', '\r'))
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "# isort:imports-FUTURE\nimport os\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.place_imports

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "import os as os\nprint('hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "from os import path as p\nfrom sys import argv as a\nprint('hello')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# special'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = "# special\nimport os\nprint('hello')"
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    var_8 = bool(var_6.import_index >= 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "# comment\nimport os\nprint('hello')"
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'above'
    var_5 = bool('above' in var_3.categorized_comments)
    assert var_5 is True



