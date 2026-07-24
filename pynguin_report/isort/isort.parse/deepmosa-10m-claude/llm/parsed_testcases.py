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
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'some code'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 'hello'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = '''hello"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'''"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = """hello'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'world"""'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello\\"world"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 5  # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 5'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path; import sys'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy; import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 5'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 'a'; y = 'b'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello"  # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello # not comment"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 'hello"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "hello'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = ()
    var_3 = module_0.skip_line(var_0, var_0, var_1, var_2)
    var_4 = bool(var_3 == (False, ''))
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# just a comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 1/3 statements.
# Partially parsed test_file_contents_simple_import. Retrieved 1/3 statements.
# Partially parsed test_file_contents_from_import. Retrieved 1/3 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 1/3 statements.
# Partially parsed test_file_contents_import_with_alias. Retrieved 1/3 statements.
# Partially parsed test_file_contents_from_import_with_alias. Retrieved 1/3 statements.
# Partially parsed test_file_contents_multiline_import_parentheses. Retrieved 1/3 statements.
# Partially parsed test_file_contents_multiline_import_backslash. Retrieved 1/3 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 1/3 statements.
# Partially parsed test_file_contents_non_import_lines. Retrieved 1/3 statements.
# Partially parsed test_file_contents_section_comment. Retrieved 1/3 statements.
# Partially parsed test_file_contents_skip_directive. Retrieved 1/3 statements.
# Partially parsed test_file_contents_semicolon_separated_statements. Retrieved 1/3 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 1/3 statements.
# Partially parsed test_file_contents_trailing_comma. Retrieved 1/3 statements.
# Partially parsed test_file_contents_cimport. Retrieved 1/3 statements.
# Partially parsed test_file_contents_line_ending_inference. Retrieved 1/3 statements.
# Partially parsed test_file_contents_with_docstring. Retrieved 1/3 statements.
# Partially parsed test_file_contents_change_count. Retrieved 1/5 statements.
# Partially parsed test_file_contents_place_imports_marker. Retrieved 1/3 statements.
# Partially parsed test_file_contents_relative_import. Retrieved 1/3 statements.
# Partially parsed test_file_contents_star_import. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = 'os'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'import os as operating_system\n'
    var_1 = 'os'
    var_2 = 'operating_system'

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = 'os.path'
    var_2 = 'p'

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = 'os'
    var_2 = 'path'
    var_3 = 'environ'

def test_case_0():
    var_0 = 'from os import \\\n    path, \\\n    environ\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'import os  # operating system\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'import os\n\nx = 1\n'
    var_1 = 'x = 1'

def test_case_0():
    var_0 = '# isort: split\nimport os\n'

def test_case_0():
    var_0 = 'import os  # isort: skip\nimport sys\n'
    var_1 = 'os'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = 'os'
    var_2 = 'sys'

def test_case_0():
    var_0 = 'from os import path as p  # comment\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'

def test_case_0():
    var_0 = '"""\nModule docstring\n"""\nimport os\n'
    var_1 = 'os'

def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'

def test_case_0():
    var_0 = 'from . import module\n'

def test_case_0():
    var_0 = 'from os import *\n'
    var_1 = 'os'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 11/21 statements.


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
    var_9 = 'os'
    var_10 = 'STDLIB'
    var_11 = {}
    var_12 = 'straight'
    var_13 = {}
    var_14 = {}



# Parsed testcases at query #4
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
    var_0 = 'x = 5'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = '# import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, getcwd'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_from_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 7/9 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_import_with_comment. Retrieved 7/9 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 5/6 statements.
# Partially parsed test_file_contents_place_imports. Retrieved 4/5 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [''])
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = bool(var_3.lines_without_imports == [])
    assert var_7 is True
    var_8 = var_3.import_index
    assert var_8 == -1
    var_9 = var_3.change_count
    assert var_9 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = 'from'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'
    var_11 = 'sys'

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
    var_6 = 'numpy'
    var_7 = bool('numpy' in var_3.as_map['straight'])
    assert var_7 is True
    var_8 = 'np'
    var_9 = bool('np' in var_3.as_map['straight']['numpy'])
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path as p\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'os.path'
    var_7 = bool('os.path' in var_3.as_map['from'])
    assert var_7 is True
    var_8 = 'p'
    var_9 = bool('p' in var_3.as_map['from']['os.path'])
    assert var_9 is True

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
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = 'from'
    var_9 = {}
    var_10 = 'os'

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
    var_6 = 'STDLIB'
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'x = 1\ny = 2\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == -1
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n\nx = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = len(var_6)
    var_8 = bool(var_7 >= 1)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    var_5 = bool(var_3.line_separator in ('\n', '\r\n', '\r'))
    assert var_5 is True

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
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: split'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\n# isort: split\nimport sys\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    var_8 = bool(var_6.import_index >= 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort:skip\nimport sys\n'
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
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""\nModule docstring\n"""\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

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
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'tests'
    var_8 = bool('tests' in var_6.imports)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n# isort:imports-THIRDPARTY\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.place_imports

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
    var_2 = 'import os\r\nimport sys\r\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\rimport sys\r'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r'

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
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

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
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

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
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

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

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_335_evaluates_to_true. Retrieved 5/7 statements.


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
    var_6 = len(var_4)
    var_7 = bool(var_6 > 0)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_trailing_comma. Retrieved 4/7 statements.
# Partially parsed test_file_contents_return_type. Retrieved 3/7 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    var_6 = bool(var_3.import_index >= 0)
    assert var_6 is True
    var_7 = var_3.imports
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    var_6 = bool(var_3.import_index >= 0)
    assert var_6 is True
    var_7 = var_3.imports
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os  # system module\nimport sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.categorized_comments
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    var_6 = bool(var_3.import_index >= 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'straight'
    var_6 = var_3.as_map[var_5]
    var_7 = len(var_6)
    var_8 = 0
    var_9 = var_7 > var_8
    var_10 = 'from'
    var_11 = var_3.as_map[var_10]
    var_12 = len(var_11)
    var_13 = var_12 > var_8
    var_14 = bool(var_9 or var_13)
    assert var_14 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    assert var_5 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'x = 1\ny = 2\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    assert var_5 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    var_6 = bool(var_3.import_index >= 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.import_index
    var_6 = bool(var_3.import_index >= 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.line_separator
    assert var_5 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import path, getcwd\n'
    var_1 = True
    var_2 = 'force_single_line'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5.import_index
    var_8 = bool(var_5.import_index >= 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = var_3.original_line_count
    var_7 = var_5 - var_6
    var_8 = var_3.change_count
    var_9 = bool(var_3.change_count == var_7)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import (\n    path,\n)\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.trailing_commas

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\nimport numpy\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = var_3.sections
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)



# Parsed testcases at query #8
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
    var_0 = 'x = 1\ny = 2\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2

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
    var_3 = var_1.import_index
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'

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
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

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
    var_0 = 'import os, \\\n    sys\n'
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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.change_count
    var_3 = bool(var_1.change_count >= 0)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something as alias'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = True
    var_6 = 'remove_redundant_aliases'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = 'import module.something as something'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = bool(var_10 is not None)
    assert var_11 is True
    var_12 = 'remove_redundant_aliases'
    var_13 = {var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from module import something as something'
    var_16 = module_1.file_contents(var_15, var_14)
    var_17 = bool(var_16 is not None)
    assert var_17 is True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from unknown_module import something\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_199_evaluates_to_true. Retrieved 9/14 statements.
# Partially parsed test_predicate_at_line_199_with_cimport. Retrieved 9/14 statements.
# Partially parsed test_predicate_at_line_199_with_line_starting_import. Retrieved 9/14 statements.
# Partially parsed test_predicate_at_line_199_with_line_starting_cimport. Retrieved 9/14 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 199 evaluates to True when import_string ends with ' import' or ' cimport'."
    var_1 = 'from module import'
    var_2 = 'something'
    var_3 = ' import'
    var_4 = ' cimport'
    var_5 = (var_3, var_4)
    var_6 = 'import '
    var_7 = 'cimport '
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = "Test that the predicate at line 199 evaluates to True when import_string ends with ' cimport'."
    var_1 = 'from module cimport'
    var_2 = 'something'
    var_3 = ' import'
    var_4 = ' cimport'
    var_5 = (var_3, var_4)
    var_6 = 'import '
    var_7 = 'cimport '
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = "Test that the predicate at line 199 evaluates to True when line starts with 'import '."
    var_1 = 'something'
    var_2 = 'import module'
    var_3 = ' import'
    var_4 = ' cimport'
    var_5 = (var_3, var_4)
    var_6 = 'import '
    var_7 = 'cimport '
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = "Test that the predicate at line 199 evaluates to True when line starts with 'cimport '."
    var_1 = 'something'
    var_2 = 'cimport module'
    var_3 = ' import'
    var_4 = ' cimport'
    var_5 = (var_3, var_4)
    var_6 = 'import '
    var_7 = 'cimport '
    var_8 = (var_6, var_7)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_from_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 9/13 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_import_with_comment. Retrieved 7/9 statements.
# Partially parsed test_file_contents_escaped_newline_import. Retrieved 7/9 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 9/13 statements.
# Partially parsed test_file_contents_in_quote. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [''])
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
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'
    var_10 = var_3.change_count
    assert var_10 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'
    var_10 = {}
    var_11 = {}
    var_12 = 'sys'

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
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'from'
    var_8 = {}
    var_9 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'x = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == -1
    var_5 = 'x = 1'
    var_6 = bool('x = 1' in var_3.lines_without_imports)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nx = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'x = 1'
    var_6 = bool('x = 1' in var_3.lines_without_imports)
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
    var_2 = 'import os\r\nimport sys\r\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    assert var_4 == 3

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
    var_2 = 'import os, \\\n    sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'STDLIB'
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'
    var_10 = {}
    var_11 = {}
    var_12 = 'sys'

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'x = 1\ny = 2\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.change_count
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
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1\nimport os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""\nimport os\n"""\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'STDLIB'
    var_5 = {}
    var_6 = 'straight'
    var_7 = {}
    var_8 = 'sys'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,  # path module\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

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
    var_8 = var_7.import_index
    assert var_8 == 0



# Parsed testcases at query #13
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
    var_5 = var_3.imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = var_3.original_line_count
    assert var_8 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.original_line_count
    assert var_5 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.original_line_count
    assert var_5 == 3

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os as operating_system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'operating_system'
    var_6 = bool('operating_system' in var_3.as_map['straight']['os'])
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
    var_2 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.in_lines
    var_6 = len(var_5)
    assert var_6 == 5

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# This is a comment\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 1
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort: skip\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'import os  # isort: skip'
    var_5 = bool('import os  # isort: skip' in var_3.lines_without_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = "import os\n\nprint('hello')\n"
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
    var_2 = 'import os\r\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\r\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import \\\n    path\n'
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
    var_2 = 'from os import path,\n'
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
    var_7 = 'straight'
    var_8 = var_5.as_map[var_7][var_6]
    var_9 = len(var_8)
    assert var_9 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# Custom section'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# Custom section\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path  # comment\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0

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
    var_2 = 'import os as o, sys as s\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

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
    var_16 = 'change_count'
    var_17 = hasattr(var_3, var_16)
    var_18 = bool(var_17)
    assert var_18 is True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 160 evaluates to True.'
    var_1 = 'from module import ('
    var_2 = '('
    var_3 = 0
    var_4 = '#'
    var_5 = var_1.split(var_4)[var_3]
    var_6 = var_2 in var_5
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 3/5 statements.


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
    var_9 = var_3.place_imports
    var_10 = bool(var_3.place_imports == {})
    assert var_10 is True
    var_11 = []
    var_12 = var_3.imports
    var_13 = var_3.change_count
    assert var_13 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.imports
    var_7 = str(var_6)
    var_8 = 'os'
    var_9 = bool('os' in var_7)
    assert var_9 is True
    var_10 = var_3.change_count
    assert var_10 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.change_count
    assert var_6 == -1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.change_count
    assert var_6 == -2

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
    var_8 = bool('print' in var_3.lines_without_imports[1])
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
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines[-1]
    assert var_4 == ''
    var_5 = var_3.in_lines
    var_6 = len(var_5)
    assert var_6 == 2

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
    var_16 = 'change_count'
    var_17 = hasattr(var_3, var_16)
    var_18 = bool(var_17)
    assert var_18 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '\r\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:skip_file\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os as operating_system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'os'
    var_7 = bool('os' in var_3.as_map['straight'])
    assert var_7 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
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
    var_0 = 'from os import path'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import (\n    func1,\n    func2\n)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module func1 func2'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import func1 \\\n    func2'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module func1 func2'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc cimport stdlib'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'libc stdlib'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import _import'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module _import'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module cimport _cimport'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module _cimport'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import { func1, func2 }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module {|func1 func2|}'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import (module1, module2, module3)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module1 module2 module3'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from package.subpackage import (func1, func2, func3)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'package.subpackage func1 func2 func3'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'from'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import func1, func2 \\ func3'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module func1 func2 func3'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 1/3 statements.
# Partially parsed test_file_contents_no_imports. Retrieved 1/7 statements.
# Partially parsed test_file_contents_simple_import. Retrieved 2/7 statements.
# Partially parsed test_file_contents_from_import. Retrieved 1/3 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 1/5 statements.
# Partially parsed test_file_contents_with_newline_ending. Retrieved 1/3 statements.
# Partially parsed test_file_contents_with_carriage_return. Retrieved 1/3 statements.
# Partially parsed test_file_contents_preserves_non_imports. Retrieved 1/5 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 1/3 statements.
# Partially parsed test_file_contents_import_with_alias. Retrieved 4/12 statements.
# Partially parsed test_file_contents_import_with_comment. Retrieved 1/3 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 1/3 statements.
# Partially parsed test_file_contents_backslash_continuation. Retrieved 1/3 statements.
# Partially parsed test_file_contents_returns_parsed_content. Retrieved 15/31 statements.
# Partially parsed test_file_contents_section_comment. Retrieved 1/5 statements.
# Partially parsed test_file_contents_change_count. Retrieved 1/7 statements.
# Partially parsed test_file_contents_line_separator_inference. Retrieved 1/3 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 1/3 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 1/5 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = "print('hello')\nprint('world')"

def test_case_0():
    var_0 = "import os\nprint('hello')"
    var_1 = 'os'

def test_case_0():
    var_0 = "from os import path\nprint('hello')"

def test_case_0():
    var_0 = 'import os\nimport sys\nfrom pathlib import Path'

def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = 'import os\r'

def test_case_0():
    var_0 = "# Comment\nprint('hello')\n# Another comment"

def test_case_0():
    var_0 = 'from os import (\n    path,\n    getcwd\n)'

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p'
    var_1 = 'straight'
    var_2 = 0
    var_3 = 'from'

def test_case_0():
    var_0 = "import os  # operating system\nprint('hello')"

def test_case_0():
    var_0 = 'import os; import sys'

def test_case_0():
    var_0 = 'from os import \\\n    path'

def test_case_0():
    var_0 = 'import os'
    var_1 = 'in_lines'
    var_2 = 'lines_without_imports'
    var_3 = 'import_index'
    var_4 = 'place_imports'
    var_5 = 'import_placements'
    var_6 = 'as_map'
    var_7 = 'imports'
    var_8 = 'categorized_comments'
    var_9 = 'change_count'
    var_10 = 'original_line_count'
    var_11 = 'line_separator'
    var_12 = 'sections'
    var_13 = 'verbose_output'
    var_14 = 'trailing_commas'

def test_case_0():
    var_0 = '# isort: split\nimport os'

def test_case_0():
    var_0 = "import os\nimport sys\nprint('hello')"

def test_case_0():
    var_0 = 'import os\nimport sys'

def test_case_0():
    var_0 = 'from os import (\n    path,  # path module\n    getcwd  # get current directory\n)'

def test_case_0():
    var_0 = 'from os import (\n    path,\n)'



# Parsed testcases at query #3
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 1 evaluates to False.'
    var_1 = ''
    var_2 = module_0.file_contents(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = 'as_found'
    var_6 = hasattr(var_3, var_5)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




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
    var_6 = bool(var_1.imports == {})
    assert var_6 is True
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nx = 1\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = "print('hello')"
    var_4 = bool("print('hello')" in var_1.lines_without_imports)
    assert var_4 is True
    var_5 = var_1.imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == -2

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv\nimport json\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

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
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system\nfrom sys import argv as args\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.as_map
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = '"""\nModule docstring.\n"""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,  # path comment\n    environ  # environ comment\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.categorized_comments
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = str(var_2)
    var_4 = 'isort:skip'
    var_5 = bool('isort:skip' in var_3)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 0
    var_3 = f'{var_1.section_comments[var_2]}\nimport os\n'
    var_4 = module_1.file_contents(var_3, var_1)
    var_5 = var_4.import_index
    var_6 = bool(var_4.import_index >= 0)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nimport sys\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = var_1.in_lines[0]
    assert var_5 == 'import os'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 6/10 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [''])
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
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = 'straight'
    var_7 = 0
    var_8 = {}
    var_9 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.imports
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.change_count
    assert var_6 == 0

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
    var_2 = 'from os import (\n    path,\n    environ\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.in_lines
    var_7 = len(var_6)
    assert var_7 == 4

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
    var_6 = 'straight'
    var_7 = var_3.as_map[var_6]
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n\ndef foo():\n    pass\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True
    var_6 = var_3.lines_without_imports
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

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
    var_2 = 'import os\r\nimport sys\r\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r\n'

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'x = 1\ny = 2\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == -1
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from . import module\n'
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
    var_2 = 'from os import (path)\n'
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
    var_6 = 'import os\n'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.import_index
    var_9 = bool(var_7.import_index >= 0)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.change_count
    var_5 = bool(var_3.change_count >= 0)
    assert var_5 is True
    var_6 = var_3.original_line_count
    var_7 = bool(var_3.original_line_count > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from libc.stdlib cimport malloc\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

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
    var_7 = bool(var_5.import_index >= 0)
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
    var_6 = var_5.import_index
    var_7 = bool(var_5.import_index >= 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.place_imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort:skip\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (  # isort:skip\n    path\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""Module docstring"""\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index > 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from __future__ import annotations\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import *\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index >= 0)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




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
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['x = 1', 'y = 2', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['x = 1', 'y = 2'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.change_count
    assert var_7 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['from'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_4 is True
    var_5 = 'sys'
    var_6 = bool('sys' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_6 is True

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
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['from'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # system module\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_4 is True

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
    var_0 = '# isort: future'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# isort: future\nimport __future__\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' not in var_1.imports[var_1.sections[0]]['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports[var_1.sections[0]]['straight'])
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
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports[var_1.sections[0]]['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports[var_1.sections[0]]['from'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.change_count
    assert var_2 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-FUTURE\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'FUTURE'
    var_3 = bool('FUTURE' in var_1.place_imports)
    assert var_3 is True

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
    var_8 = var_7.import_index
    assert var_8 == 0

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
    var_7 = bool('os' in var_5.imports[var_5.sections[0]]['from'])
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
    assert var_6 == 0



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# This is a comment\nfrom module import something\n'
    var_1 = False
    var_2 = []
    var_3 = 'treat_all_comments_as_code'
    var_4 = 'treat_comments_as_code'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1.file_contents(var_0, var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.import_index
    var_10 = len(var_9)
    var_11 = bool(var_10 >= 0)
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_195_evaluates_to_false. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 195 evaluates to False.'
    var_1 = 'module as alias'
    var_2 = 'module something'
    var_3 = ' '
    var_4 = ' as '
    var_5 = ''



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 361 (import_from not in root) evaluates to False.'
    var_1 = 'from module import something\n'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import something\nfrom module import another\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = 'import_index'
    var_8 = hasattr(var_5, var_7)
    var_9 = bool(var_8)
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 396 evaluates to True when 'isort:imports-' is in last."
    var_1 = '# Some comment with isort:imports- directive'
    var_2 = 'isort:imports-'
    var_3 = var_2 not in var_1
    assert var_3 is False
    var_4 = '# Some regular comment'
    var_5 = var_2 not in var_4
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 1/3 statements.
# Partially parsed test_file_contents_section_comments. Retrieved 1/3 statements.
# Partially parsed test_file_contents_imports_structure. Retrieved 5/6 statements.
# Partially parsed test_file_contents_verbose_output. Retrieved 1/5 statements.
# Partially parsed test_file_contents_place_imports_dict. Retrieved 3/4 statements.
# Partially parsed test_file_contents_trailing_commas. Retrieved 3/4 statements.


def test_case_0():
    var_0 = ''

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nprint('world')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.change_count
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "from os import path\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nimport sys\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True
    var_5 = 'sys'
    var_6 = bool('sys' in var_1.imports)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os  # operating system\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = bool(var_1.categorized_comments is not None)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "from os import (\n    path,\n    environ\n)\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import numpy as np\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'numpy'
    var_4 = bool('numpy' in var_1.as_map['straight'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "from os import path as p\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.as_map['from'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.in_lines[-1]
    assert var_3 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

def test_case_0():
    var_0 = '# isort: split\nimport os'

import isort.parse as module_0

def test_case_0():
    var_0 = "from os import \\\n    path\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os; import sys\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\ny = 2'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 >= 2)
    assert var_4 is True
    var_5 = var_1.import_index
    assert var_5 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'straight'
    var_3 = bool('straight' in var_1.as_map)
    assert var_3 is True
    var_4 = 'from'
    var_5 = bool('from' in var_1.as_map)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'from'
    var_3 = bool('from' in var_1.categorized_comments)
    assert var_3 is True
    var_4 = 'straight'
    var_5 = bool('straight' in var_1.categorized_comments)
    assert var_5 is True
    var_6 = 'nested'
    var_7 = bool('nested' in var_1.categorized_comments)
    assert var_7 is True
    var_8 = 'above'
    var_9 = bool('above' in var_1.categorized_comments)
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import os\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    var_4 = var_1.original_line_count
    var_5 = var_3 - var_4
    var_6 = var_1.change_count
    var_7 = bool(var_1.change_count == var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 'import os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas



# Parsed testcases at query #13
#--------------------------

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
    var_2 = 'import os\nimport sys\n'
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
    var_2 = 'import os\n\ndef hello():\n    pass\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    environ\n)\n'
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
    var_2 = 'import os as operating_system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 'straight'
    var_6 = var_3.as_map[var_5]
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
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
    var_5 = 'from'
    var_6 = var_3.as_map[var_5]
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
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
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'def hello():\n    pass\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == -1
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 2

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort:skip\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'import os  # isort:skip'
    var_5 = bool('import os  # isort:skip' in var_3.lines_without_imports)
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
    var_2 = 'from os import \\\n    path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = len(var_4)
    var_6 = var_3.original_line_count
    var_7 = bool(var_3.original_line_count == var_5)
    assert var_7 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.sections
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 8/10 statements.
# Partially parsed test_file_contents_from_import. Retrieved 8/10 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 8/10 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 8/10 statements.
# Partially parsed test_file_contents_import_with_comment. Retrieved 8/10 statements.
# Partially parsed test_file_contents_change_count. Retrieved 5/8 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 11/15 statements.
# Partially parsed test_file_contents_backslash_continuation. Retrieved 8/10 statements.
# Partially parsed test_file_contents_nested_comments. Retrieved 5/6 statements.
# Partially parsed test_file_contents_multiple_as_imports. Retrieved 10/12 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = ''
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.in_lines
    var_5 = bool(var_3.in_lines == [''])
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == -1
    var_7 = var_3.change_count
    assert var_7 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'x = 1\ny = 2\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == -1
    var_5 = var_3.lines_without_imports
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_3.lines_without_imports[0]
    assert var_7 == 'x = 1'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'from'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'
    var_11 = 'sys'

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
    var_7 = 'p'
    var_8 = bool('p' in var_3.as_map['from']['os.path'])
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import (\n    path,\n    sep\n)\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'from'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # operating system\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'straight'
    var_9 = {}
    var_10 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nx = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\r\nx = 1\r\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.line_separator
    assert var_4 == '\r\n'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# isort: THIRDPARTY'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# isort: THIRDPARTY\nimport requests\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os  # isort:skip\nx = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'x = 1'
    var_5 = bool('x = 1' in var_3.lines_without_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\nx = 1\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.lines_without_imports
    var_5 = len(var_4)
    var_6 = var_3.change_count

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nx = 1\ny = 2\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.original_line_count
    assert var_4 == 3

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os; import sys\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 0
    var_5 = var_1.default_sections[var_4]
    var_6 = {}
    var_7 = 'straight'
    var_8 = {}
    var_9 = 'os'
    var_10 = var_1.default_sections[var_4]
    var_11 = {}
    var_12 = {}
    var_13 = 'sys'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import \\\n    path\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 0
    var_5 = 0
    var_6 = var_1.default_sections[var_5]
    var_7 = {}
    var_8 = 'from'
    var_9 = {}
    var_10 = 'os'

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
    var_2 = 'from os import path  # path comment\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'nested'
    var_5 = {}
    var_6 = 'os'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""Module docstring"""\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 1

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '"""\nMultiline\nstring\n"""\nimport os\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.import_index
    assert var_4 == 4

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:imports-THIRDPARTY\nimport requests\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'THIRDPARTY'
    var_5 = bool('THIRDPARTY' in var_3.place_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort: imports-THIRDPARTY\nimport requests\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'THIRDPARTY'
    var_5 = bool('THIRDPARTY' in var_3.place_imports)
    assert var_5 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os as o, sys as s\n'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'straight'
    var_5 = var_3.as_map[var_4]
    var_6 = 'os'
    var_7 = []
    var_8 = 'o'
    var_9 = var_3.as_map[var_4]
    var_10 = 'sys'
    var_11 = []
    var_12 = 's'

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path, sep\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_259_evaluates_to_true. Retrieved 5/9 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import nested_module as alias  # comment for alias\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True
    var_5 = '__iter__'
    var_6 = hasattr(var_3, var_5)



# Parsed testcases at query #16
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = "Test that the predicate at line 199 evaluates to True when import_string ends with ' import' or ' cimport'."
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
    var_10 = 'from os import (\n    path,\n    environ\n)'
    var_11 = module_0.file_contents(var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True
    var_13 = 'from libc.stdlib cimport (\n    malloc,\n    free\n)'
    var_14 = module_0.file_contents(var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #17
#--------------------------

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
    assert var_2 == -1
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
    assert var_5 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

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
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.original_line_count
    assert var_2 == 2
    var_3 = var_1.lines_without_imports[-1]
    assert var_3 == ''

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
    var_0 = 'from os import path  # comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.categorized_comments['from'])
    assert var_4 is True

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
    var_0 = 'import os  # isort:skip\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (  # isort:skip\n    path\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# STDLIB'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# STDLIB\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

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
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    var_3 = bool(var_1.line_separator in ('\r\n', '\n', '\r'))
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring"""\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports
    var_3 = len(var_2)
    var_4 = bool(var_3 >= 0)
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.trailing_commas
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 0)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 428 evaluates to False.\n    \n    The predicate is: if placed_module and placed_module not in imports\n    This should evaluate to False when:\n    1. placed_module is empty string (falsy), OR\n    2. placed_module exists in imports dict\n    '
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'import os'
    var_4 = module_1.file_contents(var_3, var_2)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = 'os'
    var_7 = [var_6]
    var_8 = 'known_standard_library'
    var_9 = {var_8: var_7}
    var_10 = module_0.Config(**var_9)
    var_11 = 'import os\n'
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = bool(var_12 is not None)
    assert var_13 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_placed_module_equals_empty_string. Retrieved 9/19 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'sections'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = 'import unknown_module_that_wont_be_placed\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = ''
    var_12 = bool('' in var_10.imports)
    assert var_12 is True
    var_13 = 'straight'
    var_14 = bool('straight' in var_10.imports[''])
    assert var_14 is True
    var_15 = 'unknown_module_that_wont_be_placed'
    var_16 = bool('unknown_module_that_wont_be_placed' in var_10.imports['']['straight'])
    assert var_16 is True



# Parsed testcases at query #20
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
    var_0 = 'import os  # operating system\n'
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

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nprint(x)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['x = 1', 'print(x)', ''])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n\ndef foo():\n    pass\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'os'
    var_4 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_4 is True
    var_5 = var_1.change_count
    assert var_5 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == [])
    assert var_4 is True

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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# Custom section'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# Custom section\nimport os\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['from'])
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.
# Partially parsed test_file_contents_returns_parsed_content. Retrieved 8/11 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = var_1.change_count
    assert var_6 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv\nimport json\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # operating system\nimport sys  # system\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.categorized_comments
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'straight'
    var_4 = var_1.as_map[var_3]
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_5 > var_6
    var_8 = 'from'
    var_9 = var_1.as_map[var_8]
    var_10 = len(var_9)
    var_11 = var_10 > var_6
    var_12 = bool(var_7 or var_11)
    assert var_12 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nx = 5\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n\ndef foo():\n    pass\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

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
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = '\n\n\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'

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
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os, \\\n    sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports
    var_3 = len(var_2)
    var_4 = 0
    var_5 = var_3 >= var_4
    var_6 = bool('THIRDPARTY' in var_1.place_imports or var_5)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'in_lines'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'imports'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'import_index'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to True when line is in section_comments.'
    var_1 = '# isort: split'
    var_2 = '# Custom section'
    var_3 = [var_1, var_2]
    var_4 = 'section_comments'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = '# isort: split\nimport os'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to True when line is in section_comments_end.'
    var_1 = '# end imports'
    var_2 = [var_1]
    var_3 = 'section_comments_end'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = "import os\n# end imports\nprint('hello')"
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to True when line is in either section_comments or section_comments_end.'
    var_1 = '# isort: start'
    var_2 = [var_1]
    var_3 = '# isort: end'
    var_4 = [var_3]
    var_5 = 'section_comments'
    var_6 = 'section_comments_end'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = "# isort: start\nimport os\nimport sys\n# isort: end\nprint('hello')"
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = bool(var_10 is not None)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_file_contents_verbose_output. Retrieved 5/8 statements.
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
    var_5 = var_1.imports
    var_6 = bool(var_1.imports == {})
    assert var_6 is True
    var_7 = var_1.change_count
    assert var_7 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nx = 1\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_1.imports
    var_6 = bool(var_1.imports == {})
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = var_1.change_count
    var_7 = bool(var_1.change_count >= 0)
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import argv\nimport json\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = var_1.original_line_count
    assert var_6 == 3

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'straight'
    var_4 = var_1.as_map[var_3]
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_5 > var_6
    var_8 = 'from'
    var_9 = var_1.as_map[var_8]
    var_10 = len(var_9)
    var_11 = var_10 > var_6
    var_12 = bool(var_7 or var_11)
    assert var_12 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
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
    var_0 = 'import os\r\nimport sys\r\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.line_separator
    var_4 = bool(var_1.line_separator in ('\r\n', '\n', '\r'))
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.trailing_commas
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, \\\n    environ\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n\nx = 1\nprint(x)\n'
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
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = True
    var_2 = 'force_single_line'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.import_index
    assert var_6 == 0
    var_7 = var_5.imports
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: split\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    var_3 = bool(var_1.import_index >= 0)
    assert var_3 is True

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
    var_0 = '# isort:imports-FUTURE\nimport os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path as p, environ as e\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = 'from'
    var_4 = var_1.as_map[var_3]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



