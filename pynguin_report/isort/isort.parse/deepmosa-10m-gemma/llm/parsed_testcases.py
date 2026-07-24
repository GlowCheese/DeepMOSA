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
    var_0 = '"""hello"""'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""hello"""'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("it\\"s fine")'
    var_1 = '"'
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
    var_0 = 'cimport cython; import os'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

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
    var_0 = 'import os; # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'single'"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print(\'"\')'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 18/29 statements.
# Partially parsed test_file_contents_empty_input. Retrieved 14/17 statements.


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
    var_9 = True
    var_10 = False
    var_11 = False
    var_12 = []
    var_13 = False
    var_14 = False
    var_15 = 'os'
    var_16 = 'STDLIB'
    var_17 = 'THIRDPARTY'
    var_18 = lambda x: var_16 if x == var_15 else var_17
    var_19 = "import os\nimport sys\n\nprint('hello')"
    var_20 = 'import os'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '\n'
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = True
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = []
    var_11 = False
    var_12 = False
    var_13 = ''



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
    var_6 = 'import math  '
    var_7 = module_1.import_type(var_6, var_3)
    assert var_7 == 'straight'
    var_8 = 'cimport some_module'
    var_9 = module_1.import_type(var_8, var_3)
    assert var_9 == 'straight'

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
    var_6 = 'from datetime import datetime'
    var_7 = module_1.import_type(var_6, var_3)
    assert var_7 == 'from'

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
    var_6 = 'from os import path # NOQA'
    var_7 = module_1.import_type(var_6, var_3)
    assert var_7 is None

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
    var_4 = 'x = 10'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None
    var_6 = "print('hello')"
    var_7 = module_1.import_type(var_6, var_3)
    assert var_7 is None
    var_8 = '  import os'
    var_9 = module_1.import_type(var_8, var_3)
    assert var_9 is None



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = ''
    var_1 = []
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
    var_13 = module_1.file_contents(var_0, var_12)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_predicate_true. Retrieved 9/10 statements.


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
    var_13 = '# isort:imports-MAIN\nimport os'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = 'MAIN'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 1/31 statements.
# Partially parsed test_file_contents_structure_assertion. Retrieved 29/33 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'FIRST'
    var_1 = 'SECOND'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = {}
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = {}
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = {var_0: var_6, var_1: var_9}
    var_11 = 'import os'
    var_12 = [var_11]
    var_13 = []
    var_14 = 0
    var_15 = {}
    var_16 = {}
    var_17 = {}
    var_18 = {}
    var_19 = {var_2: var_17, var_3: var_18}
    var_20 = {}
    var_21 = -1
    var_22 = 1
    var_23 = '\n'
    var_24 = [var_0, var_1]
    var_25 = []
    var_26 = set()
    var_27 = []
    var_28 = 'in_lines'
    var_29 = 'lines_without_imports'
    var_30 = 'import_index'
    var_31 = 'place_imports'
    var_32 = 'import_placements'
    var_33 = 'as_map'
    var_34 = 'imports'
    var_35 = 'categorized_comments'
    var_36 = 'change_count'
    var_37 = 'original_line_count'
    var_38 = 'line_separator'
    var_39 = 'sections'
    var_40 = 'verbose_output'
    var_41 = 'trailing_commas'
    var_42 = {var_28: var_12, var_29: var_13, var_30: var_14, var_31: var_15, var_32: var_16, var_33: var_19, var_34: var_10, var_35: var_20, var_36: var_21, var_37: var_22, var_38: var_23, var_39: var_24, var_40: var_25, var_41: var_26}
    var_43 = module_0.ParsedContent(*var_27, **var_42)
    var_44 = var_43.in_lines
    var_45 = bool(var_43.in_lines == ['import os'])
    assert var_45 is True
    var_46 = var_43.imports['FIRST']
    var_47 = bool(var_43.imports['FIRST'] == {'straight': {}, 'from': {}})
    assert var_47 is True
    var_48 = var_43.original_line_count
    assert var_48 == 1
    var_49 = var_43.trailing_commas



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 37/51 statements.


def test_case_0():
    var_0 = 'STDLIB'
    var_1 = 'FIRSTPARTY'
    var_2 = 'THIRDPARTY'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = '\n'
    var_6 = []
    var_7 = []
    var_8 = True
    var_9 = True
    var_10 = False
    var_11 = False
    var_12 = False
    var_13 = []
    var_14 = False
    var_15 = False
    var_16 = 'import os\nimport requests\nfrom my_local_module import func\n'
    var_17 = 'import os'
    var_18 = 'import requests'
    var_19 = 'from my_local_module import func'
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_17, var_18, var_19]
    var_22 = 3
    var_23 = {}
    var_24 = {}
    var_25 = 'straight'
    var_26 = 'from'
    var_27 = {}
    var_28 = {}
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = {}
    var_31 = {}
    var_32 = 0
    var_33 = '\n'
    var_34 = 'STDLIB'
    var_35 = 'FIRSTPARTY'
    var_36 = 'THIRDPARTY'
    var_37 = [var_34, var_35, var_36]
    var_38 = []
    var_39 = set()
    var_40 = 'STDLIB'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_import_type_is_truthy. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'import os'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_line_separator_from_config. Retrieved 2/7 statements.
# Partially parsed test_file_contents_line_separator_from_inference. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'import os'
    var_1 = 'inferred'

def test_case_0():
    var_0 = 'import os'
    var_1 = '\n'



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = set()
    var_5 = 'sections'
    var_6 = 'forced_separate'
    var_7 = 'treat_all_comments_as_code'
    var_8 = 'treat_comments_as_code'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = 'import os\n# comment\nimport sys'
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_file_contents_predicate_false_by_empty_out_lines. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'main'
    var_1 = -1
    var_2 = []
    var_3 = len(var_2)
    var_4 = 1
    var_5 = max(var_1, var_4, var_4)
    var_6 = var_5 - var_4
    var_7 = var_3 > var_6
    assert var_7 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_line_separator_from_config. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'import os\nimport sys'



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
    var_0 = 'from os import path'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path'

import isort.parse as module_0

def test_case_0():
    var_0 = '_cimport my_module'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == '_cimport my_module'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, name)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path name'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\npath'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from module import { func }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module {|func|}'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from my_package import _import_module, (sub_module)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'my_package _import_module sub_module'

import isort.parse as module_0

def test_case_0():
    var_0 = 'just_a_string'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'just_a_string'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 18/71 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'mock_module'
    var_3 = 'place'
    var_4 = 'STDLIB'
    var_5 = False
    var_6 = ''
    var_7 = (var_5, var_6)
    var_8 = 'import os'
    var_9 = (var_8, var_8)
    var_10 = 'straight'
    var_11 = None
    var_12 = (var_11, var_11)
    var_13 = lambda x: x
    var_14 = lambda x: x
    var_15 = dict()
    var_16 = lambda k: var_15
    var_17 = 'import os\nimport sys'
    var_18 = module_1.file_contents(var_17, var_1)
    var_19 = 'STDLIB'
    var_20 = bool('STDLIB' in var_18.imports)
    assert var_20 is True
    var_21 = var_18.imports['STDLIB']['straight']['os']
    assert var_21 is True
    var_22 = var_18.original_line_count
    assert var_22 == 2



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = None
    var_4 = False
    var_5 = False
    var_6 = 'main'
    var_7 = 'straight'
    var_8 = 'from'
    var_9 = {}
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'os'
    var_14 = 'extra_section'
    var_15 = bool(var_14 and var_14 not in var_12)
    assert var_15 is True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = 'sections'
    var_6 = 'forced_separate'
    var_7 = 'section_comments'
    var_8 = 'section_comments_end'
    var_9 = 'float_to_top'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = '\n'
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = 'line_ending'
    var_18 = 'sections'
    var_19 = 'forced_separate'
    var_20 = 'section_comments'
    var_21 = 'section_comments_end'
    var_22 = 'float_to_top'
    var_23 = {var_17: var_12, var_18: var_13, var_19: var_14, var_20: var_15, var_21: var_16, var_22: var_4}
    var_24 = module_0.Config(**var_23)
    var_25 = 'import os'
    var_26 = bool(var_24.line_ending is not None and var_24.line_ending != '')
    assert var_26 is True



# Parsed testcases at query #5
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
    var_7 = 'from os import path # some comment\n'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = var_8.imports['main']['from']['path']
    assert var_9 == 'some comment'



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'main'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = False
    var_5 = set()
    var_6 = 'sections'
    var_7 = 'forced_separate'
    var_8 = 'force_single_line'
    var_9 = 'treat_all_comments_as_code'
    var_10 = 'treat_comments_as_code'
    var_11 = {var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.Config(**var_11)
    var_13 = '# some comment\nfrom os import path'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 4/37 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = "import os\nimport requests\nfrom math import sqrt\n\nprint('hello')"
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'module'
    var_4 = 'import os\nimport sys\nfrom datetime import datetime'
    var_5 = 'os'
    var_6 = 'sys'
    var_7 = 'datetime'



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from os import path'
    var_3 = 'from math import pi'
    var_4 = module_1.file_contents(var_3, var_1)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'float_to_top'
    var_6 = 'section_comments'
    var_7 = 'section_comments_end'
    var_8 = 'forced_separate'
    var_9 = 'sections'
    var_10 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3, var_9: var_4}
    var_11 = module_0.Config(**var_10)
    var_12 = 'x = 1'
    var_13 = module_1.file_contents(var_12, var_11)
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_file_contents_basic_parsing. Retrieved 14/63 statements.


def test_case_0():
    var_0 = 'place'
    var_1 = ''
    var_2 = (var_1, var_1)
    var_3 = 'from'
    var_4 = 'import'
    var_5 = 'straight'
    var_6 = None
    var_7 = 0
    var_8 = '#'
    var_9 = 1
    var_10 = '('
    var_11 = ')'
    var_12 = '\n'
    var_13 = 'import os\nfrom requests import get\nimport my_local_module'
    var_14 = 'STDLIB'
    var_15 = 'os'
    var_16 = 'requests'
    var_17 = 'get'
    var_18 = 'my_local_module'



