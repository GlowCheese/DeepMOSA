####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = []
    var_4 = 'line_number'
    var_5 = 'indented'
    var_6 = 'module'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Import(*var_3, **var_7)
    var_9 = str(var_8)
    assert var_9 == '1 import os'

import isort.identify as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'numpy'
    var_3 = 'np'
    var_4 = []
    var_5 = 'line_number'
    var_6 = 'indented'
    var_7 = 'module'
    var_8 = 'alias'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Import(*var_4, **var_9)
    var_11 = str(var_10)
    assert var_11 == '5 import numpy as np'

import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 'math'
    var_3 = 'sqrt'
    var_4 = []
    var_5 = 'line_number'
    var_6 = 'indented'
    var_7 = 'module'
    var_8 = 'attribute'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Import(*var_4, **var_9)
    var_11 = str(var_10)
    assert var_11 == '10 from math import sqrt'

import isort.identify as module_0

def test_case_0():
    var_0 = 2
    var_1 = True
    var_2 = 'cython'
    var_3 = []
    var_4 = 'line_number'
    var_5 = 'indented'
    var_6 = 'module'
    var_7 = 'cimport'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_1}
    var_9 = module_0.Import(*var_3, **var_8)
    var_10 = str(var_9)
    assert var_10 == '2 indented cimport cython'

import pathlib as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 15
    var_1 = False
    var_2 = 'sys'
    var_3 = '/src/main.py'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = []
    var_8 = 'line_number'
    var_9 = 'indented'
    var_10 = 'module'
    var_11 = 'file_path'
    var_12 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_6}
    var_13 = module_1.Import(*var_7, **var_12)
    var_14 = str(var_13)
    assert var_14 == '/src/main.py:15 import sys'

import pathlib as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'sklearn.svm'
    var_3 = 'SVC'
    var_4 = 'SVC_model'
    var_5 = 'lib/utils.py'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = []
    var_10 = 'line_number'
    var_11 = 'indented'
    var_12 = 'module'
    var_13 = 'attribute'
    var_14 = 'alias'
    var_15 = 'file_path'
    var_16 = {var_10: var_0, var_11: var_1, var_12: var_2, var_13: var_3, var_14: var_4, var_15: var_8}
    var_17 = module_1.Import(*var_9, **var_16)
    var_18 = str(var_17)
    assert var_18 == 'lib/utils.py:20 indented from sklearn.svm import SVC as SVC_model'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 1/14 statements.
# Partially parsed test_imports_from_import. Retrieved 1/13 statements.
# Partially parsed test_imports_with_as_alias. Retrieved 1/12 statements.
# Partially parsed test_imports_with_as_from_import. Retrieved 1/13 statements.
# Partially parsed test_imports_multiline_parentheses. Retrieved 1/12 statements.
# Partially parsed test_imports_ignores_comments_and_indented_logic. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'

def test_case_0():
    var_0 = 'from os import path, name'

def test_case_0():
    var_0 = 'import numpy as np'

def test_case_0():
    var_0 = 'from os import path as p'

def test_case_0():
    var_0 = 'from os import (\n    path,\n    name\n)'

def test_case_0():
    var_0 = '# This is a comment\nimport os  # end of line comment'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_imports_predicate_line_1_is_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'import os\n'
    var_1 = '__iter__'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_imports_predicate_line_16_is_false_via_top_only. Retrieved 4/9 statements.
# Partially parsed test_imports_predicate_line_16_is_false_via_not_matching_declarations. Retrieved 5/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'def my_function():\n    import os\n'
    var_1 = ()
    var_2 = False
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = ()
    var_2 = False
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_imports_predicate_evaluates_to_true. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = True
    var_2 = ()
    var_3 = 'remove_redundant_aliases'
    var_4 = 'section_comments'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_imports_predicate_false_when_top_only_is_true_and_line_is_declaration. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'def'
    var_1 = 'class'
    var_2 = 'def my_function():\n    import os\n'
    var_3 = False
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_imports_simple_import. Retrieved 4/8 statements.
# Partially parsed test_imports_from_import. Retrieved 4/8 statements.
# Partially parsed test_imports_with_as_alias. Retrieved 4/8 statements.
# Partially parsed test_imports_skipping_non_import_lines. Retrieved 4/8 statements.
# Partially parsed test_imports_handling_cimport. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path, name\nfrom sys import argv'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom os import path as p'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'x = 1\nimport math\nprint(x)'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'cimport mymodule'
    var_1 = True
    var_2 = ()
    var_3 = 'semicolon_split'
    var_4 = 'section_comments'
    var_5 = 'remove_redundant_aliases'
    var_6 = {var_3: var_1, var_4: var_2, var_5: var_1}
    var_7 = module_0.Config(**var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_imports_basic_straight_import. Retrieved 6/10 statements.
# Partially parsed test_imports_from_import. Retrieved 6/10 statements.
# Partially parsed test_imports_with_as_alias. Retrieved 6/10 statements.
# Partially parsed test_imports_cimport. Retrieved 5/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = list(var_1)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[0].module
    assert var_9 == 'os'
    var_10 = var_7[1].module
    assert var_10 == 'sys'
    var_11 = var_7[0].index
    assert var_11 == 1
    var_12 = var_7[1].index
    assert var_12 == 2

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path, name'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = list(var_1)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_7[0].module
    assert var_9 == 'os'
    var_10 = var_7[0].attribute
    assert var_10 == 'path'
    var_11 = var_7[1].module
    assert var_11 == 'os'
    var_12 = var_7[1].attribute
    assert var_12 == 'name'

import isort.settings as module_0

def test_case_0():
    var_0 = 'import os as system_os'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = list(var_1)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[0].module
    assert var_9 == 'os'
    var_10 = var_7[0].alias
    assert var_10 == 'system_os'

import isort.settings as module_0

def test_case_0():
    var_0 = 'cimport math'
    var_1 = ()
    var_2 = 'section_comments'
    var_3 = 'remove_redundant_aliases'
    var_4 = {var_2: var_1, var_3: var_0}
    var_5 = module_0.Config(**var_4)
    var_6 = list(var_1)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[0].module
    assert var_8 == 'math'
    var_9 = var_6[0].cimport
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_imports_top_only_predicate_false. Retrieved 1/11 statements.
# Partially parsed test_imports_predicate_logic_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'import os\n'

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'import os'
    var_3 = 'def'
    var_4 = 'class'
    var_5 = (var_3, var_4)
    var_6 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_imports_top_only_false_predicate_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_imports_returns_iterator. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'import os\nfrom sys import argv\n'
    var_1 = '__iter__'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_imports_predicate_line_1_evaluates_to_true. Retrieved 13/43 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'isort_module'
    var_1 = {}
    var_2 = 'def '
    var_3 = 'class '
    var_4 = 'async def '
    var_5 = lambda cls, idx, ind, cimport=False, file_path=None, module=None, attribute=None, alias=None: cls(idx, ind, cimport=cimport, file_path=file_path, module=module, attribute=attribute, alias=alias)
    var_6 = 'line1'
    var_7 = ''
    var_8 = (var_6, var_7)
    var_9 = 'import os'
    var_10 = (var_9, var_7)
    var_11 = (var_9, var_9)
    var_12 = 'import os\n'
    var_13 = {}
    var_14 = module_0.Config(**var_13)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_imports_predicate_false_when_top_only_is_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = "Ensures that the predicate 'top_only and not in_quote and raw_line.startswith(STATEMENT_DECLARATIONS)'\n    evaluates to False when top_only is False.\n    "
    var_1 = 'from __future__ import print_function\nimport os\n'
    var_2 = False
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_imports_simple_import_statement. Retrieved 4/8 statements.
# Partially parsed test_imports_from_import_statement. Retrieved 4/8 statements.
# Partially parsed test_imports_with_alias. Retrieved 4/8 statements.
# Partially parsed test_imports_with_from_and_alias. Retrieved 4/8 statements.
# Partially parsed test_imports_skips_comments_and_strings. Retrieved 4/8 statements.
# Partially parsed test_imports_handles_multiline_parentheses. Retrieved 4/8 statements.
# Partially parsed test_imports_handles_backslash_line_continuation. Retrieved 4/8 statements.
# Partially parsed test_imports_cimport_detection. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path, name\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import numpy as np\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path as p\n'
    var_1 = None
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os  # comment\n"import hidden"\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    name\n)\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import os, \\\n    sys\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'cimport math\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_imports_basic_import_straight. Retrieved 1/10 statements.
# Partially parsed test_imports_from_import. Retrieved 1/10 statements.
# Partially parsed test_imports_with_as_alias. Retrieved 1/10 statements.
# Partially parsed test_imports_with_as_alias_from_import. Retrieved 1/10 statements.
# Partially parsed test_imports_skipping_comments_and_quotes. Retrieved 1/10 statements.
# Partially parsed test_imports_with_cimport. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n'

def test_case_0():
    var_0 = 'from os import path, name\n'

def test_case_0():
    var_0 = 'import numpy as np\n'

def test_case_0():
    var_0 = 'from os import path as p\n'

def test_case_0():
    var_0 = '# This is a comment\nimport os  # inline comment\n"""Multi-line\nstring"""\nimport sys\n'

def test_case_0():
    var_0 = 'cimport math\n'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.identify as module_0

def test_case_0():
    var_0 = 1
    var_1 = False
    var_2 = 'os'
    var_3 = []
    var_4 = 'line_number'
    var_5 = 'indented'
    var_6 = 'module'
    var_7 = {var_4: var_0, var_5: var_1, var_6: var_2}
    var_8 = module_0.Import(*var_3, **var_7)
    var_9 = str(var_8)
    assert var_9 == '1 import os'

import isort.identify as module_0

def test_case_0():
    var_0 = 10
    var_1 = False
    var_2 = 'numpy'
    var_3 = 'np'
    var_4 = []
    var_5 = 'line_number'
    var_6 = 'indented'
    var_7 = 'module'
    var_8 = 'alias'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Import(*var_4, **var_9)
    var_11 = str(var_10)
    assert var_11 == '10 import numpy as np'

import isort.identify as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'math'
    var_3 = 'sqrt'
    var_4 = []
    var_5 = 'line_number'
    var_6 = 'indented'
    var_7 = 'module'
    var_8 = 'attribute'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Import(*var_4, **var_9)
    var_11 = str(var_10)
    assert var_11 == '5 from math import sqrt'

import isort.identify as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'math'
    var_3 = 'sqrt'
    var_4 = 's'
    var_5 = []
    var_6 = 'line_number'
    var_7 = 'indented'
    var_8 = 'module'
    var_9 = 'attribute'
    var_10 = 'alias'
    var_11 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_3, var_10: var_4}
    var_12 = module_0.Import(*var_5, **var_11)
    var_13 = str(var_12)
    assert var_13 == '5 from math import sqrt as s'

import isort.identify as module_0

def test_case_0():
    var_0 = 2
    var_1 = False
    var_2 = 'libc'
    var_3 = True
    var_4 = []
    var_5 = 'line_number'
    var_6 = 'indented'
    var_7 = 'module'
    var_8 = 'cimport'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Import(*var_4, **var_9)
    var_11 = str(var_10)
    assert var_11 == '2 cimport libc'

import pathlib as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 20
    var_1 = True
    var_2 = 'sys'
    var_3 = 'src/main.py'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = []
    var_8 = 'line_number'
    var_9 = 'indented'
    var_10 = 'module'
    var_11 = 'file_path'
    var_12 = {var_8: var_0, var_9: var_1, var_10: var_2, var_11: var_6}
    var_13 = module_1.Import(*var_7, **var_12)
    var_14 = str(var_13)
    assert var_14 == 'src/main.py:20 indented import sys'

import pathlib as module_0
import isort.identify as module_1

def test_case_0():
    var_0 = 100
    var_1 = True
    var_2 = 'tensorflow'
    var_3 = 'layers'
    var_4 = 'tf_layers'
    var_5 = False
    var_6 = 'lib/utils.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = []
    var_11 = 'line_number'
    var_12 = 'indented'
    var_13 = 'module'
    var_14 = 'attribute'
    var_15 = 'alias'
    var_16 = 'cimport'
    var_17 = 'file_path'
    var_18 = {var_11: var_0, var_12: var_1, var_13: var_2, var_14: var_3, var_15: var_4, var_16: var_5, var_17: var_9}
    var_19 = module_1.Import(*var_10, **var_18)
    var_20 = str(var_19)
    assert var_20 == 'lib/utils.py:100 indented from tensorflow import layers as tf_layers'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_imports_simple_import_statement. Retrieved 4/8 statements.
# Partially parsed test_imports_from_import_statement. Retrieved 4/8 statements.
# Partially parsed test_imports_with_alias. Retrieved 4/8 statements.
# Partially parsed test_imports_with_as_in_from_import. Retrieved 4/8 statements.
# Partially parsed test_imports_skipping_yield_statement. Retrieved 4/8 statements.
# Partially parsed test_imports_with_comments. Retrieved 4/8 statements.
# Partially parsed test_imports_multiline_import_with_parentheses. Retrieved 4/8 statements.
# Partially parsed test_imports_with_cimport. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path, name'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import pandas as pd'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import path as p'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'yield\nimport math'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'import os  # system os\nimport sys # system sys'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    name\n)'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = 'cimport sys'
    var_1 = []
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_imports_predicate_line_1_is_true. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import os'
    var_2 = ''
    var_3 = (var_1, var_2)
    var_4 = (var_1, var_1)
    var_5 = 'def'
    var_6 = 'class'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_imports_predicate_at_line_1_is_true. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'import os\n'
    var_1 = 'import os'
    var_2 = (var_1, var_1)
    var_3 = ''
    var_4 = (var_1, var_3)
    var_5 = (var_1, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_imports_predicate_evaluates_to_true. Retrieved 4/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'import os\n'
    var_1 = ()
    var_2 = True
    var_3 = 'section_comments'
    var_4 = 'remove_redundant_aliases'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)



