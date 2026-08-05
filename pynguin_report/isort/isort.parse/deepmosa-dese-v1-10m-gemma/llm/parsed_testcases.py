####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "world'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'hello"'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = '"""start'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "'''start"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "'quote' end"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "'it\\'s me'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from math import sin'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; # x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "';' "
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 14/29 statements.
# Partially parsed test_file_contents_structure. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = '\n'
    var_5 = []
    var_6 = []
    var_7 = False
    var_8 = True
    var_9 = False
    var_10 = False
    var_11 = False
    var_12 = set()
    var_13 = False
    var_14 = False
    var_15 = 'import os\nimport requests\n'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = []
    var_3 = '\n'
    var_4 = []
    var_5 = []
    var_6 = False
    var_7 = True
    var_8 = set()
    var_9 = 'import os\n'



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0.Config()
    var_4 = "import os\nprint('hello')"
    var_5 = module_1.file_contents(var_4, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_file_contents_trigger_placed_module_empty. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'standard'
    var_1 = ''
    var_2 = 'import unknown_module'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'x = 1\n'
    var_7 = module_1.file_contents(var_6, var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_predicate_false_via_no_import_type. Retrieved 2/9 statements.
# Partially parsed test_file_contents_import_index_not_minus_one. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'import os\nfrom sys import path'
    var_1 = 'from os import (\n    path\n)\nimport sys'

def test_case_0():
    var_0 = 'import (\n    os\n)\nimport sys'



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'imports'
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0.Config()
    var_4 = 'from os import path # some comment\n'
    var_5 = module_1.file_contents(var_4, var_3)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0.Config()
    var_4 = 'from os import path, name'
    var_5 = module_1.file_contents(var_4, var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_predicate_false_no_just_imports. Retrieved 2/9 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = module_0.Config()
    var_5 = 'import os as system_os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'import'
    var_8 = 'os'
    var_9 = 'as'
    var_10 = 'system_os'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = '{|'
    var_13 = '{ '
    var_14 = '|}'
    var_15 = ' }'
    var_16 = [item.replace(var_9, var_10).replace(var_14, var_15) for item in var_11]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_file_contents_predicate_at_line_273. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'straight'
    var_1 = 'import os as system_os\n# some comment'
    var_2 = '# some comment'
    var_3 = [var_2]
    var_4 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_import_index_not_minus_one. Retrieved 5/15 statements.
# Partially parsed test_file_contents_import_index_not_minus_one_direct. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'main'
    var_1 = 'import os\nimport sys'
    var_2 = 'import math\nfrom os import path'
    var_3 = 'import os\nimport sys'
    var_4 = 'import os\nimport sys'

def test_case_0():
    var_0 = 'main'
    var_1 = 'import os\nimport sys'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_contents_predicate_false_by_type. Retrieved 1/7 statements.
# Partially parsed test_file_contents_predicate_false_by_parts_length. Retrieved 1/7 statements.
# Partially parsed test_file_contents_predicate_false_by_no_comments. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = 'from os import path # comment\n'

def test_case_0():
    var_0 = 'from os import path\n'



# Parsed testcases at query #14
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = module_0.Config()
    var_5 = 'from unknown_module import something'
    var_6 = 'main'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_0.Config()
    var_10 = module_1.file_contents(var_5, var_9)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from math import sqrt'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'math sqrt'

import isort.parse as module_0

def test_case_0():
    var_0 = '_cimport my_module'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == '_cimport my_module'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, name'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path name'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import (func1, func2)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module func1 func2'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import \\\n  submodule'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module submodule'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import my_long_module_name'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'my_long_module_name'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import { func }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module {|func|}'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import join, exists'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os.path join exists'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 15/32 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = '\n'
    var_5 = []
    var_6 = []
    var_7 = False
    var_8 = True
    var_9 = False
    var_10 = False
    var_11 = False
    var_12 = []
    var_13 = False
    var_14 = False
    var_15 = 'place'
    var_16 = 'import os\nfrom requests import get\n\ndef my_func():\n    pass'



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = '# some comment'
    var_1 = [var_0]
    var_2 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 21/69 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRD_PARTY'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = '\n'
    var_5 = []
    var_6 = []
    var_7 = True
    var_8 = True
    var_9 = False
    var_10 = False
    var_11 = False
    var_12 = []
    var_13 = False
    var_14 = False
    var_15 = 'place'
    var_16 = '\n'
    var_17 = lambda x: var_16
    var_18 = None
    var_19 = lambda msg, stacklevel: var_18
    var_20 = module_0.Config()
    var_21 = 'import os\nfrom requests import get\n'
    var_22 = module_1.file_contents(var_21, var_20)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = set()
    var_5 = None
    var_6 = False
    var_7 = False
    var_8 = 'module1, module2'
    var_9 = 'module2'
    var_10 = [var_9]
    var_11 = 'module1'
    var_12 = [var_11]
    var_13 = 'module1, module2'
    var_14 = -1
    var_15 = -1
    var_16 = var_12[var_15]
    var_17 = import_string.split(var_16)[var_14]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_predicate_true. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'import os'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'cimport mymodule'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'from os import path'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 == 'from'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'x = 1'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'import os  # noqa'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'import os  # noqa'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'import os  # isort:skip'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 is None
    var_4 = 'import os  # isort: skip'
    var_5 = module_1.import_type(var_4, var_1)
    assert var_5 is None
    var_6 = 'import os  # isort: split'
    var_7 = module_1.import_type(var_6, var_1)
    assert var_7 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Config()
    var_2 = 'import os # NOQA'
    var_3 = module_1.import_type(var_2, var_1)
    assert var_3 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 19/57 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'os'
    var_3 = 'requests'
    var_4 = ''
    var_5 = lambda x: var_0 if x == var_2 else var_1 if x == var_3 else var_4
    var_6 = 'place'
    var_7 = lambda x: (x, x)
    var_8 = 'from'
    var_9 = 'import'
    var_10 = 'straight'
    var_11 = lambda x, cfg: var_8 if var_8 in x else var_10 if var_9 in x else var_4
    var_12 = False
    var_13 = (var_12, var_4)
    var_14 = None
    var_15 = (var_14, var_14)
    var_16 = lambda x: x
    var_17 = lambda x: x
    var_18 = 'import os\nfrom requests import get'



# Parsed testcases at query #9
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = '"""text'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = "'''text"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("\\"")'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'from math import sin; import cos'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport cython; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_335_is_true. Retrieved 6/30 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'place'
    var_2 = 'from os import path\n# This is a comment'
    var_3 = False
    var_4 = module_0.Config()
    var_5 = module_1.file_contents(var_2, var_4)



# Parsed testcases at query #11
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\r\nline2'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\rline2'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\nline2'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'singleline'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\r\nline2\nline3'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_import_type_straight_import. Retrieved 6/9 statements.
# Partially parsed test_import_type_cimport. Retrieved 6/9 statements.
# Partially parsed test_import_type_from_import. Retrieved 6/9 statements.
# Partially parsed test_import_type_none_for_non_import. Retrieved 6/9 statements.
# Partially parsed test_import_type_with_noqa_and_honor_noqa_true. Retrieved 6/9 statements.
# Partially parsed test_import_type_with_noqa_and_honor_noqa_false. Retrieved 6/9 statements.
# Partially parsed test_import_type_isort_skip_detection. Retrieved 8/13 statements.
# Partially parsed test_import_type_case_insensitivity_noqa. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'import os'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'cimport sys'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'from os import path'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'x = 1'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'import os  # noqa'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'import os  # noqa'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = 'import os  # isort:skip'
    var_6 = 'from os import path  # isort: skip'
    var_7 = 'import os  # isort: split'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'import os # NOQA'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 20/32 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = '\n'
    var_5 = True
    var_6 = True
    var_7 = False
    var_8 = False
    var_9 = False
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = False
    var_14 = False
    var_15 = 'os'
    var_16 = 'STDLIB'
    var_17 = 'THIRDPARTY'
    var_18 = 'import os\nimport sys\n'
    var_19 = module_0.Config()
    var_20 = 'place'
    var_21 = module_1.file_contents(var_18, var_19)



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = module_0.Config()
    var_6 = '\r\n'
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = module_0.Config()
    var_12 = 'import os\n'
    var_13 = var_11.line_ending
    var_14 = module_1._infer_line_separator(var_12)
    var_15 = var_13 or var_14
    assert var_15 == '\r\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_file_contents_predicate_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = 2



