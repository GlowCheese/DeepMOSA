####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_file_contents_basic_import. Retrieved 15/27 statements.
# Partially parsed test_file_contents_empty_string. Retrieved 1/16 statements.
# Partially parsed test_file_contents_with_std_import. Retrieved 16/20 statements.


import isort.settings as module_0

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
    var_12 = False
    var_13 = False
    var_14 = []
    var_15 = {}
    var_16 = module_0.Config(**var_15)
    var_17 = 'import os\nimport sys\n'

def test_case_0():
    var_0 = ''

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = []
    var_3 = '\n'
    var_4 = []
    var_5 = []
    var_6 = False
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = False
    var_12 = False
    var_13 = []
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = 'import os\n'
    var_17 = module_1.file_contents(var_16, var_15)
    var_18 = 'STDLIB'
    var_19 = bool('STDLIB' in var_17.imports)
    assert var_19 is True
    var_20 = var_17.imports['STDLIB']['straight']['os']
    assert var_20 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 16/36 statements.


import isort.settings as module_0
import isort.parse as module_1

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
    var_15 = {}
    var_16 = module_0.Config(**var_15)
    var_17 = "import os\nimport requests\nprint('hello')"
    var_18 = module_1.file_contents(var_17, var_16)
    var_19 = var_18.original_line_count
    assert var_19 == 3
    var_20 = 'os'
    var_21 = bool('os' in var_18.imports['STDLIB']['straight'])
    assert var_21 is True
    var_22 = 'requests'
    var_23 = bool('requests' in var_18.imports['THIRDPARTY']['straight'])
    assert var_23 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_file_contents_predicate_true. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'main'
    var_1 = 'from os import path # This is a comment'
    var_2 = 'from'
    var_3 = 'path'



# Parsed testcases at query #4
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
    var_6 = 'verbose'
    var_7 = 'only_modified'
    var_8 = {var_6: var_0, var_7: var_0}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import os'
    var_11 = module_1.file_contents(var_10, var_9)
    var_12 = var_11.verbose_output
    var_13 = len(var_12)
    var_14 = bool(var_13 >= 0)
    assert var_14 is True



# Parsed testcases at query #5
#--------------------------




import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'import os'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 == 'straight'

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'cimport math'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 == 'straight'

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'from os import path'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 == 'from'

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'x = 1'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 is None

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'import os  # noqa'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 is None

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'import os  # noqa'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 == 'straight'

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = False
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'import os  # isort:skip'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 is None
    var_11 = 'from os import path  # isort: skip'
    var_12 = module_1.import_type(var_11, var_8)
    assert var_12 is None
    var_13 = 'import sys  # isort: split'
    var_14 = module_1.import_type(var_13, var_8)
    assert var_14 is None

import builtins as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'honor_noqa'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = {}
    var_7 = module_0.type(*var_5, **var_6)
    var_8 = var_7()
    var_9 = 'import os # NOQA'
    var_10 = module_1.import_type(var_9, var_8)
    assert var_10 is None



# Parsed testcases at query #6
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'plain text'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "text 'inside'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'start' end"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = "'quote' text"
    var_6 = ()
    var_7 = module_0.skip_line(var_5, var_1, var_2, var_6)
    var_8 = bool(var_7 == (False, ''))
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""start'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'text \\' still in quote'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; invalid_part'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from math import sqrt'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; invalid_part'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("hello") # "unclosed quote'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = ()
    var_3 = module_0.skip_line(var_0, var_0, var_1, var_2)
    var_4 = bool(var_3 == (False, ''))
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_isort_imports_predicate_true. Retrieved 10/35 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '# isort:imports-main\n'
    var_3 = 'main'
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'sections'
    var_9 = 'forced_separate'
    var_10 = 'section_comments'
    var_11 = 'section_comments_end'
    var_12 = {var_8: var_4, var_9: var_5, var_10: var_6, var_11: var_7}
    var_13 = module_0.Config(**var_12)
    var_14 = module_1.file_contents(var_2, var_13)
    var_15 = '# isort:imports-'
    var_16 = bool('# isort:imports-' in var_2)
    assert var_16 is True
    var_17 = '#'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 12/51 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'STDLIB'
    var_2 = lambda x: (x, x)
    var_3 = 'straight'
    var_4 = lambda x: x
    var_5 = None
    var_6 = lambda x: (x, var_5)
    var_7 = False
    var_8 = ''
    var_9 = (var_7, var_8)
    var_10 = lambda x: x
    var_11 = 'import os\nimport sys'
    var_12 = 'os'
    var_13 = 'sys'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_evaluates_as_predicate_true. Retrieved 14/16 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = 'sections'
    var_5 = 'forced_separate'
    var_6 = 'remove_redundant_aliases'
    var_7 = 'combine_as_imports'
    var_8 = {var_4: var_1, var_5: var_2, var_6: var_3, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import os as system\n'
    var_11 = module_1.file_contents(var_10, var_9)
    var_12 = 'as'
    var_13 = bool('as' in ['import', 'os', 'as', 'system'])
    assert var_13 is True
    var_14 = 'import'
    var_15 = 'os'
    var_16 = 'as'
    var_17 = 'system'
    var_18 = [var_14, var_15, var_16, var_17]
    var_19 = [var_14, var_15, var_16, var_17]
    var_20 = len(var_19)



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = []
    var_5 = 'sections'
    var_6 = 'forced_separate'
    var_7 = 'treat_all_comments_as_code'
    var_8 = 'treat_comments_as_code'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = 'import os\nimport sys'
    var_12 = module_1.file_contents(var_11, var_10)



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = None
    var_4 = False
    var_5 = set()
    var_6 = False
    var_7 = False
    var_8 = {}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import os\n'
    var_11 = module_1.file_contents(var_10, var_9)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'MAIN'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = False
    var_6 = 'sections'
    var_7 = 'forced_separate'
    var_8 = 'section_comments'
    var_9 = 'section_comments_end'
    var_10 = 'float_to_top'
    var_11 = {var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.Config(**var_11)
    var_13 = '# isort: imports-TOP\nimport os'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = 'isort: imports-TOP'
    var_16 = bool('isort: imports-TOP' in var_14.import_placements)
    assert var_16 is True



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = '# section start'
    var_1 = {var_0}
    var_2 = '# section end'
    var_3 = {var_2}
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '# section start\nimport os'
    var_9 = module_1.file_contents(var_8, var_7)
    var_10 = bool(var_9 is not None)
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 19/51 statements.


def test_case_0():
    var_0 = 'import os\nimport sys\n\ndef main():\n    pass'
    var_1 = 'os'
    var_2 = 'sys'
    var_3 = [var_1, var_2]
    var_4 = 'STDLIB'
    var_5 = 'THIRDPARTY'
    var_6 = 'STDLIB'
    var_7 = [var_6]
    var_8 = []
    var_9 = '\n'
    var_10 = []
    var_11 = []
    var_12 = False
    var_13 = True
    var_14 = False
    var_15 = False
    var_16 = False
    var_17 = []
    var_18 = False
    var_19 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 13/35 statements.


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
    var_10 = False
    var_11 = []
    var_12 = '# isort:section'
    var_13 = [var_12]
    var_14 = '# isort:end-section'
    var_15 = [var_14]
    var_16 = 'import os\nimport sys\nfrom datetime import datetime'
    var_17 = 'import os'
    var_18 = 'from datetime import datetime'
    var_19 = 'STDLIB'
    var_20 = 'THIRDPARTY'



# Parsed testcases at query #16
#--------------------------




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



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'from os import path # some comment'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.imports['main']['from']['import path']
    var_10 = bool(var_8.imports['main']['from']['import path'] == {'comment': '# some comment'})
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = False
    var_5 = 'sections'
    var_6 = 'forced_separate'
    var_7 = 'remove_redundant_aliases'
    var_8 = 'combine_as_imports'
    var_9 = 'verbose'
    var_10 = 'only_modified'
    var_11 = 'force_single_line'
    var_12 = 'treat_all_comments_as_code'
    var_13 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4, var_9: var_4, var_10: var_4, var_11: var_4, var_12: var_4}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from os import path\n'
    var_16 = module_1.file_contents(var_15, var_14)



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\nimport sys\nfrom datetime import datetime'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 8/32 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'place'
    var_3 = 'os'
    var_4 = 'requests'
    var_5 = ''
    var_6 = lambda x: var_0 if x == var_3 else var_1 if x == var_4 else var_5
    var_7 = 'import os\nimport requests\nx = 1'
    var_8 = 'os'
    var_9 = 'requests'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_file_contents_no_as_keyword. Retrieved 13/15 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = 'sections'
    var_5 = 'forced_separate'
    var_6 = 'remove_redundant_aliases'
    var_7 = 'combine_as_imports'
    var_8 = {var_4: var_1, var_5: var_2, var_6: var_3, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import os\nfrom math import sqrt'
    var_11 = module_1.file_contents(var_10, var_9)
    var_12 = 'import os'
    var_13 = module_1.strip_syntax(var_12)
    var_14 = '{|'
    var_15 = '{ '
    var_16 = '|}'
    var_17 = ' }'
    var_18 = 'as'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_file_contents_predicate_at_line_241_false. Retrieved 13/15 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = 'sections'
    var_5 = 'forced_separate'
    var_6 = 'remove_redundant_aliases'
    var_7 = 'combine_as_imports'
    var_8 = {var_4: var_1, var_5: var_2, var_6: var_3, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'import os\n'
    var_11 = module_1.file_contents(var_10, var_9)
    var_12 = 'import os'
    var_13 = module_1.strip_syntax(var_12)
    var_14 = '{|'
    var_15 = '{ '
    var_16 = '|}'
    var_17 = ' }'
    var_18 = 'as'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_file_contents_simple_import. Retrieved 23/69 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = 'os'
    var_2 = 'STDLIB'
    var_3 = 'THIRDPARTY'
    var_4 = lambda m, config: var_2 if m == var_1 else var_3
    var_5 = lambda l: (l, l)
    var_6 = 'from'
    var_7 = 'import'
    var_8 = 'straight'
    var_9 = ''
    var_10 = lambda l, c: var_6 if var_6 in l else var_8 if var_7 in l else var_9
    var_11 = 0
    var_12 = '#'
    var_13 = 1
    var_14 = None
    var_15 = lambda l: (l.split(var_12)[var_11].strip(), l.split(var_12)[var_13].strip() if var_12 in l else var_14)
    var_16 = lambda l: l
    var_17 = False
    var_18 = (var_17, var_9)
    var_19 = lambda l, **kwargs: var_18
    var_20 = {}
    var_21 = 'import os\nimport sys'
    var_22 = {}
    var_23 = module_0.Config(**var_22)
    var_24 = module_1.file_contents(var_21, var_23)
    var_25 = var_24.in_lines
    var_26 = bool(var_24.in_lines == ['import os', 'import sys'])
    assert var_26 is True
    var_27 = 'os'
    var_28 = bool('os' in var_24.imports['STDLIB']['straight'])
    assert var_28 is True
    var_29 = 'sys'
    var_30 = bool('sys' in var_24.imports['THIRDPARTY']['straight'])
    assert var_30 is True



# Parsed testcases at query #24
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
    var_0 = ''
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'singleline'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = '\r\n\r\n'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'text\nwith\r\nnewline'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'main'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = set()
    var_6 = 'force_single_line'
    var_7 = 'sections'
    var_8 = 'forced_separate'
    var_9 = 'verbose'
    var_10 = 'only_modified'
    var_11 = 'treat_all_comments_as_code'
    var_12 = 'treat_comments_as_code'
    var_13 = {var_6: var_0, var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_4, var_11: var_4, var_12: var_5}
    var_14 = module_0.Config(**var_13)
    var_15 = 'from os import path\n# my comment'
    var_16 = module_1.file_contents(var_15, var_14)
    var_17 = bool(True)
    assert var_17 is True



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
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring"""'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('it\\'s me')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; invalid_code'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from math import sin'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; invalid_code'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello') # comment"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'start' and 'end'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("it\'s fine")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 8/50 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'os'
    var_3 = 'requests'
    var_4 = ''
    var_5 = lambda x: var_0 if x == var_2 else var_1 if x == var_3 else var_4
    var_6 = 'import os\nimport requests\n'
    var_7 = 'import os\nfrom sys import argv\n'

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'cimport math'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 == 'from'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None

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
    var_4 = 'import os  # isort:skip'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None
    var_6 = 'import os  # isort: skip'
    var_7 = module_1.import_type(var_6, var_3)
    assert var_7 is None
    var_8 = 'import os  # isort: split'
    var_9 = module_1.import_type(var_8, var_3)
    assert var_9 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os # NOQA'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_file_contents_basic_import. Retrieved 21/37 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
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
    var_16 = 'os'
    var_17 = 'STDLIB'
    var_18 = 'THIRDPARTY'
    var_19 = lambda module, config: var_17 if module == var_16 else var_18
    var_20 = {}
    var_21 = module_0.Config(**var_20)
    var_22 = 'import os\nimport sys'
    var_23 = module_1.file_contents(var_22, var_21)
    var_24 = bool(var_15)
    assert var_24 is True
    var_25 = var_23.original_line_count
    assert var_25 == 2
    var_26 = 'os'
    var_27 = bool('os' in var_23.imports['STDLIB']['straight'])
    assert var_27 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
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
    var_15 = {}
    var_16 = module_0.Config(**var_15)
    var_17 = 'from os import path'
    var_18 = module_1.file_contents(var_17, var_16)
    var_19 = var_18.imports['STDLIB']['from']['os']['path']
    assert var_19 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = [var_0]
    var_2 = []
    var_3 = '\n'
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = []
    var_12 = False
    var_13 = False
    var_14 = {}
    var_15 = module_0.Config(**var_14)
    var_16 = ''
    var_17 = module_1.file_contents(var_16, var_15)
    var_18 = var_17.original_line_count
    assert var_18 == 0
    var_19 = var_17.lines_without_imports
    var_20 = len(var_19)
    assert var_20 == 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 15/27 statements.


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
    var_12 = False
    var_13 = False
    var_14 = []
    var_15 = 'import os\nimport sys\nfrom datetime import datetime'
    var_16 = 'import os\nfrom sys import argv'
    var_17 = 'os'
    var_18 = 'argv'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_line_226_predicate_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'imports'
    var_1 = 'from os import path'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_trailing_comma_in_import_string. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'main'
    var_1 = 'from os import path,'
    var_2 = 'from os import path,'
    var_3 = 'path'
    var_4 = [var_3]
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = bool(var_4[-1])
    assert var_6 is True
    var_7 = -1
    var_8 = -1
    var_9 = var_4[var_8]
    var_10 = var_2.split(var_9)[var_7]
    var_11 = ','
    var_12 = bool(',' in var_10)
    assert var_12 is True



# Parsed testcases at query #8
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
    var_0 = ''
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'mixed\r\nand\r'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 7/28 statements.
# Partially parsed test_file_contents_import_placements. Retrieved 2/18 statements.
# Partially parsed test_file_contents_empty_input. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRDPARTY'
    var_2 = 'os'
    var_3 = 'requests'
    var_4 = ''
    var_5 = lambda x: var_0 if x == var_2 else var_1 if x == var_3 else var_4
    var_6 = 'import os\nimport requests\nx = 1'
    var_7 = 'os'
    var_8 = 'requests'

def test_case_0():
    var_0 = 'FIRST'
    var_1 = '# isort:imports-FIRST\nimport os'
    var_2 = '# isort:imports-FIRST'

def test_case_0():
    var_0 = 'STDLIB'
    var_1 = ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_file_contents_predicate_true. Retrieved 11/13 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = 'sections'
    var_5 = 'forced_separate'
    var_6 = 'remove_redundant_aliases'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.Config(**var_7)
    var_9 = 'from .module import something\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = 'import '
    var_12 = 'from .module import something'
    var_13 = '\n'
    var_14 = ' '



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_file_contents_placed_module_is_empty. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'main'
    var_1 = 'from module import name\n'



# Parsed testcases at query #12
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "variable = 'value"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "variable = 'value'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'variable = "value'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'variable = "value"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""docstring"""'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'''docstring"
    var_1 = "'''"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'''"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'''docstring'''"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "variable = 'it\\'s me'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from math import sin'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport libc; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'cimport libc; import os'
    var_7 = ()
    var_8 = module_0.skip_line(var_6, var_1, var_2, var_7, var_4)
    var_9 = bool(var_8 == (False, ''))
    assert var_9 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "variable = 'value' # comment"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 1 # 'unclosed quote"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = 'from os import path'
    var_2 = -1
    var_3 = var_0[var_2]
    var_4 = ','
    var_5 = -1
    var_6 = -1
    var_7 = var_0[var_6]
    var_8 = var_1.split(var_7)[var_5]
    var_9 = var_4 in var_8
    var_10 = var_0 and var_3 and var_9
    var_11 = bool(var_10)
    assert var_11 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_file_contents_isort_imports_predicate_true. Retrieved 6/28 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'place'
    var_1 = []
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = '# isort:imports-MAIN\n'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_file_contents_predicate_true. Retrieved 8/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 'main'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = set()
    var_6 = True
    var_7 = 'import os\n'



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'main'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = set()
    var_6 = 'force_single_line'
    var_7 = 'sections'
    var_8 = 'forced_separate'
    var_9 = 'treat_all_comments_as_code'
    var_10 = 'treat_comments_as_code'
    var_11 = {var_6: var_0, var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.Config(**var_11)
    var_13 = 'from os import path\n# comment'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_file_contents_comments_exist_for_module. Retrieved 23/45 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'place'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'import os\n# comment\nimport sys'
    var_4 = 'os'
    var_5 = [var_4]
    var_6 = '# comment'
    var_7 = [var_6]
    var_8 = 'os'
    var_9 = 'straight'
    var_10 = 'from'
    var_11 = 'nested'
    var_12 = 'above'
    var_13 = {}
    var_14 = {}
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {var_9: var_16, var_10: var_17}
    var_19 = {var_9: var_13, var_10: var_14, var_11: var_15, var_12: var_18}
    var_20 = len(var_7)
    var_21 = 0
    var_22 = var_20 > var_21
    var_23 = bool(var_7 is not None and var_22)
    assert var_23 is True
    var_24 = []
    var_25 = var_19['straight']['os']
    var_26 = bool(var_19['straight']['os'] == ['# comment'])
    assert var_26 is True
    var_27 = bool(var_24 == [])
    assert var_27 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'THIRD_PARTY'
    var_2 = "import os\nimport sys\n\nprint('hello')"
    var_3 = 'mock_module'
    var_4 = 'os'
    var_5 = 'sys'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_428_is_true. Retrieved 8/19 statements.


import collections as module_0

def test_case_0():
    var_0 = 'main'
    var_1 = 'new_section'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = []
    var_5 = {}
    var_6 = module_0.OrderedDict(*var_4, **var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_0.OrderedDict(*var_7, **var_8)
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = {var_0: var_10}
    var_12 = bool(var_1 and var_1 not in var_11)
    assert var_12 is True



