####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




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
    var_0 = 'x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = 'test"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "test'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "test'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'test"'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "test\\'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "x = '''test"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'''"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = """test'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "test'''"
    var_1 = "'''"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'test"""'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import x; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from x import y; z = 3'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport x; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '### comment'
    var_1 = ''
    var_2 = 0
    var_3 = '###'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = \'test\'; y = "test"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "# comment 'quote"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True



# Parsed testcases at query #2
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
    var_0 = 'import os, sys'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, dirname)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path dirname'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, \\ dirname'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path dirname'

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; from sys import path; cimport numpy'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os ; sys path ; numpy'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import {path, dirname}'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os {|path dirname|}'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'os.path'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os.path'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import _os'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == '_import _os'

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport _numpy'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == '_cimport _numpy'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os.path import (dirname, basename), \\ join'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os.path dirname basename join'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/19 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'sys'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -1
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    environ,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'environ'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == -4
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == {'os'})
    assert var_36 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:imports-thirdparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:skip\nimport os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:skip', 'import os', 'import sys'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:skip', 'import os', 'import sys'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 3
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.in_lines
    var_7 = bool(var_5.in_lines == ['import os'])
    assert var_7 is True
    var_8 = var_5.lines_without_imports
    var_9 = bool(var_5.lines_without_imports == [])
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = 'from module import something'
    var_1 = 'import '
    var_2 = bool('import ' in var_0)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_line_strip_ends_with_backslash. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'import module \\'
    var_1 = '\\'



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = 'line1'
    var_1 = 'line2'
    var_2 = 'line3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 1
    var_6 = max(var_4, var_5)
    var_7 = var_6 - var_5
    var_8 = len(var_3)
    var_9 = bool(var_8 > var_7)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_file_contents_empty_input. Retrieved 5/8 statements.
# Partially parsed test_file_contents_single_line_import. Retrieved 10/17 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/21 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/17 statements.
# Partially parsed test_file_contents_with_as. Retrieved 13/19 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/22 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/17 statements.
# Partially parsed test_file_contents_with_skip_comment. Retrieved 5/8 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = var_1.imports
    var_15 = bool(var_1.imports == {})
    assert var_15 is True
    var_16 = var_1.categorized_comments
    var_17 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_17 is True
    var_18 = var_1.change_count
    assert var_18 == 0
    var_19 = var_1.original_line_count
    assert var_19 == 1
    var_20 = var_1.line_separator
    assert var_20 == '\n'
    var_21 = var_1.sections
    var_22 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_22 is True
    var_23 = var_1.verbose_output
    var_24 = bool(var_1.verbose_output == [])
    assert var_24 is True
    var_25 = set()
    var_26 = var_1.trailing_commas
    var_27 = bool(var_1.trailing_commas == var_25)
    assert var_27 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'STDLIB'
    var_15 = 'os'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = []
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -1
    var_25 = var_1.original_line_count
    assert var_25 == 1
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_28 is True
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import argv'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'STDLIB'
    var_15 = []
    var_16 = 'sys'
    var_17 = 'argv'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os  # comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'STDLIB'
    var_15 = 'os'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = []
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -1
    var_25 = var_1.original_line_count
    assert var_25 == 1
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_28 is True
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'numpy'
    var_14 = 'np'
    var_15 = [var_14]
    var_16 = {var_13: var_15}
    var_17 = var_1.as_map
    var_18 = 'THIRDPARTY'
    var_19 = False
    var_20 = (var_13, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = []
    var_24 = var_1.imports
    var_25 = var_1.categorized_comments
    var_26 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_26 is True
    var_27 = var_1.change_count
    assert var_27 == -1
    var_28 = var_1.original_line_count
    assert var_28 == 1
    var_29 = var_1.line_separator
    assert var_29 == '\n'
    var_30 = var_1.sections
    var_31 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_31 is True
    var_32 = var_1.verbose_output
    var_33 = bool(var_1.verbose_output == [])
    assert var_33 is True
    var_34 = set()
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == var_34)
    assert var_36 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import (\n    argv,\n    path,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import (', '    argv,', '    path,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'STDLIB'
    var_15 = []
    var_16 = 'sys'
    var_17 = 'argv'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = 'path'
    var_21 = (var_20, var_18)
    var_22 = [var_19, var_21]
    var_23 = [var_22]
    var_24 = var_1.imports
    var_25 = var_1.categorized_comments
    var_26 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_26 is True
    var_27 = var_1.change_count
    assert var_27 == -4
    var_28 = var_1.original_line_count
    assert var_28 == 4
    var_29 = var_1.line_separator
    assert var_29 == '\n'
    var_30 = var_1.sections
    var_31 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_31 is True
    var_32 = var_1.verbose_output
    var_33 = bool(var_1.verbose_output == [])
    assert var_33 is True
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == {'sys'})
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:imports-thirdparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'THIRDPARTY'
    var_15 = 'numpy'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = []
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -2
    var_25 = var_1.original_line_count
    assert var_25 == 2
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_28 is True
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:skip\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:skip', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:skip', 'import os'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = var_1.imports
    var_15 = bool(var_1.imports == {})
    assert var_15 is True
    var_16 = var_1.categorized_comments
    var_17 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_17 is True
    var_18 = var_1.change_count
    assert var_18 == 0
    var_19 = var_1.original_line_count
    assert var_19 == 2
    var_20 = var_1.line_separator
    assert var_20 == '\n'
    var_21 = var_1.sections
    var_22 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_22 is True
    var_23 = var_1.verbose_output
    var_24 = bool(var_1.verbose_output == [])
    assert var_24 is True
    var_25 = set()
    var_26 = var_1.trailing_commas
    var_27 = bool(var_1.trailing_commas == var_25)
    assert var_27 is True

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = "print('hello')\nimport os"
    var_1 = True
    var_2 = 'float_to_top'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_144_evaluates_to_false. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'from'
    var_1 = 'module as alias'
    var_2 = None
    var_3 = 'from'
    var_4 = var_0 == var_3
    var_5 = ' '
    var_6 = ' as '
    var_7 = ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 12/16 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_nested_import. Retrieved 12/19 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == -1
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == 0
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'import sys'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = 'sys'
    var_20 = (var_19, var_17)
    var_21 = [var_18, var_20]
    var_22 = [var_21]
    var_23 = []
    var_24 = var_1.imports
    var_25 = var_1.categorized_comments
    var_26 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_26 is True
    var_27 = var_1.change_count
    assert var_27 == 0
    var_28 = var_1.original_line_count
    assert var_28 == 2
    var_29 = var_1.line_separator
    assert var_29 == '\n'
    var_30 = var_1.sections
    var_31 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_31 is True
    var_32 = var_1.verbose_output
    var_33 = bool(var_1.verbose_output == [])
    assert var_33 is True
    var_34 = set()
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == var_34)
    assert var_36 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os as operating_system'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'os': ['operating_system']}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sys,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    sys,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'sys'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == 0
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == {'os'})
    assert var_36 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_verbose_output_else_type_place_module. Retrieved 9/13 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os\nimport sys'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = var_7.verbose_output
    var_9 = 'else-type place_module for os returned '
    var_10 = var_7.verbose_output
    var_11 = 'else-type place_module for sys returned '



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = '# Comment'
    var_1 = 'import module'
    var_2 = [var_0, var_1]
    var_3 = 'above'
    var_4 = 'straight'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'module'
    var_9 = 0
    var_10 = 2
    var_11 = False
    var_12 = []
    var_13 = 'treat_all_comments_as_code'
    var_14 = 'treat_comments_as_code'
    var_15 = {var_13: var_11, var_14: var_12}
    var_16 = module_0.Config(**var_15)
    var_17 = bool(var_2)
    var_18 = bool(var_17)
    assert var_18 is True



# Parsed testcases at query #12
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
    var_4 = 'import sys  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = False
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import sys  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import sys  # isort:skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import sys  # isort:split'
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
    var_0 = '   '
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import sys  # NOQA'
    var_5 = module_1.import_type(var_4)
    assert var_5 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_content. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 12/19 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.
# Failed to parse test_file_contents_with_skip_comment.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 0
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == 0
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\nfrom sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'x = 1', 'from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['x = 1'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = 'sys'
    var_22 = 'path'
    var_23 = (var_22, var_17)
    var_24 = [var_23]
    var_25 = [var_24]
    var_26 = var_1.imports
    var_27 = var_1.categorized_comments
    var_28 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_28 is True
    var_29 = var_1.change_count
    assert var_29 == 0
    var_30 = var_1.original_line_count
    assert var_30 == 3
    var_31 = var_1.line_separator
    assert var_31 == '\n'
    var_32 = var_1.sections
    var_33 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_33 is True
    var_34 = var_1.verbose_output
    var_35 = bool(var_1.verbose_output == [])
    assert var_35 is True
    var_36 = set()
    var_37 = var_1.trailing_commas
    var_38 = bool(var_1.trailing_commas == var_36)
    assert var_38 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    sep,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'sep'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == 0
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == {'os'})
    assert var_36 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:imports-thirdparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'SECTION'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = 'SECTION'
    var_10 = bool('SECTION' in var_8.imports)
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_404_evaluates_to_true. Retrieved 26/34 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '# Comment'
    var_1 = 'import module'
    var_2 = [var_0, var_1]
    var_3 = 'above'
    var_4 = 'straight'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 0
    var_9 = 1
    var_10 = 'module'
    var_11 = False
    var_12 = []
    var_13 = 'treat_all_comments_as_code'
    var_14 = 'treat_comments_as_code'
    var_15 = {var_13: var_11, var_14: var_12}
    var_16 = module_0.Config(**var_15)
    var_17 = -1
    var_18 = var_2[var_17]
    var_19 = ''
    var_20 = 'straight'
    var_21 = 'above'
    var_22 = var_7[var_21][var_20]
    var_23 = []
    var_24 = 0
    var_25 = -1
    var_26 = -1
    var_27 = var_2[var_26]
    var_28 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_imports. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_alias. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 10/17 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -1
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nfrom sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = 'sys'
    var_22 = 'path'
    var_23 = (var_22, var_17)
    var_24 = [var_23]
    var_25 = [var_24]
    var_26 = var_1.imports
    var_27 = var_1.categorized_comments
    var_28 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_28 is True
    var_29 = var_1.change_count
    assert var_29 == -2
    var_30 = var_1.original_line_count
    assert var_30 == 2
    var_31 = var_1.line_separator
    assert var_31 == '\n'
    var_32 = var_1.sections
    var_33 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_33 is True
    var_34 = var_1.verbose_output
    var_35 = bool(var_1.verbose_output == [])
    assert var_35 is True
    var_36 = set()
    var_37 = var_1.trailing_commas
    var_38 = bool(var_1.trailing_commas == var_36)
    assert var_38 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os  # comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -2
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -3
    var_27 = var_1.original_line_count
    assert var_27 == 3
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == {'os'})
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['x = 1', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['x = 1'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 9/19 statements.


def test_case_0():
    var_0 = '# This is a comment'
    var_1 = 'import module'
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = var_2[var_3]
    var_5 = ''
    var_6 = '#'
    var_7 = '"""'
    var_8 = "'''"



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_392. Retrieved 8/13 statements.


def test_case_0():
    var_0 = '# Comment'
    var_1 = 'import module'
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = var_2[var_3]
    var_5 = '#'
    var_6 = '"""'
    var_7 = "'''"
    var_8 = 'isort:imports-'
    var_9 = 'isort: imports-'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_line_in_section_comments_or_end. Retrieved 8/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '# Section 1'
    var_1 = [var_0]
    var_2 = '# End Section 1'
    var_3 = [var_2]
    var_4 = 'section_comments'
    var_5 = 'section_comments_end'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = module_0.Config(**var_6)
    var_8 = '# Section 1\nimport os\n# End Section 1'
    var_9 = '# Section 1'
    var_10 = False
    var_11 = bool((var_9 in var_7.section_comments or var_9 in var_7.section_comments_end) and (not var_10))
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, sys'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc cimport printf'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'libc printf'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, from libc cimport printf'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path libc printf'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (path, sys)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\\, sys'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os _import path'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os _import path'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from libc _cimport printf'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'libc _cimport printf'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import { path, sys }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os {| path sys |}'

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'os path sys'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os path sys'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from import cimport'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os _import (path\\, sys), from libc _cimport { printf }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'os _import path sys libc _cimport {| printf |}'



# Parsed testcases at query #2
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {'os': True}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'sys': ['path']}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {}, 'from': {'sys': {'path': True}}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'import sys'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {'os': True, 'sys': True}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 2
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {'os': True}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 2
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'straight': {'numpy': True}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 2
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import (\n    path,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import (', '    path,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'sys': ['path']}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {}, 'from': {'sys': {'path': True}}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 3
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == {'sys'})
    assert var_25 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'straight': {'numpy': True}, 'from': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import path  # comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'sys': ['path']}})
    assert var_12 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = []
    var_3 = bool(var_1 and var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #4
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

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path\nfrom sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'path'
    var_3 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_3 is True
    var_4 = 'argv'
    var_5 = bool('argv' in var_1.imports['STDLIB']['from']['sys'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom pandas import DataFrame as df'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'np'
    var_3 = bool('np' in var_1.as_map['straight']['numpy'])
    assert var_3 is True
    var_4 = 'df'
    var_5 = bool('df' in var_1.as_map['from']['pandas.DataFrame'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # Operating system\n# Comment above\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'Operating system'
    var_3 = bool('Operating system' in var_1.categorized_comments['straight']['os'])
    assert var_3 is True
    var_4 = '# Comment above'
    var_5 = bool('# Comment above' in var_1.categorized_comments['above']['straight']['sys'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ,\n)\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'path'
    var_3 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_3 is True
    var_4 = 'environ'
    var_5 = bool('environ' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = 'sys'
    var_7 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports-thirdparty\nimport numpy\nimport pandas'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'pandas'
    var_5 = bool('pandas' in var_1.imports['THIRDPARTY']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort: skip\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'import os  # isort: skip'
    var_3 = bool('import os  # isort: skip' in var_1.lines_without_imports)
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.lines_without_imports
    var_4 = len(var_3)
    assert var_4 == 0

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path, environ,\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path  # Path module\nfrom sys import argv  # Argument list'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'Path module'
    var_3 = bool('Path module' in var_1.categorized_comments['nested']['os']['path'])
    assert var_3 is True
    var_4 = 'Argument list'
    var_5 = bool('Argument list' in var_1.categorized_comments['nested']['sys']['argv'])
    assert var_5 is True

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



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import (\n    # comment1\n    func1,\n    # comment2\n    func2,\n)'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_428_evaluates_to_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'straight'
    var_2 = 'from'
    var_3 = []
    var_4 = []
    var_5 = 'section1'



# Parsed testcases at query #7
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
    var_6 = 'from module import something'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = bool(var_7 is not None)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_alias. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/19 statements.
# Partially parsed test_file_contents_with_escaped_newline. Retrieved 11/18 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from collections import defaultdict'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'collections'
    var_18 = 'defaultdict'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == 0
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import sys'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'sys'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort: imports-thirdparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort: imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from typing import (\n    List,\n    Dict,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from typing import (', '    List,', '    Dict,', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'typing'
    var_18 = 'List'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'Dict'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == 0
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == {'typing'})
    assert var_36 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import \\\n    path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import \\', '    path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == 0
    var_27 = var_1.original_line_count
    assert var_27 == 2
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import sys; import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import sys; import os'])
    assert var_3 is True
    var_4 = bool(var_1)
    assert var_4 is True



# Parsed testcases at query #9
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
    var_0 = 'import os  # noqa'
    var_1 = True
    var_2 = 'honor_noqa'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.import_type(var_0, var_4)
    assert var_5 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
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

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os  # NOQA'
    var_1 = True
    var_2 = 'honor_noqa'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.import_type(var_0, var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os  # noqa'
    var_1 = False
    var_2 = 'honor_noqa'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.import_type(var_0, var_4)
    assert var_5 == 'straight'

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os  # noqa  '
    var_1 = True
    var_2 = 'honor_noqa'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.import_type(var_0, var_4)
    assert var_5 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_142_evaluates_to_false. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'from'
    var_1 = 'module as alias'
    var_2 = None
    var_3 = 'from'
    var_4 = var_0 == var_3
    var_5 = ' '
    var_6 = ' as '
    var_7 = ''



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'This is a comment'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_file_contents_single_line_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_alias. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/19 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == [])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == [])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'sys'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -1
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == [])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == 0
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == [])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == [])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    environ', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'environ'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == -4
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == [])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == {'os'})
    assert var_36 is True



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_module'
    var_5 = 'test_module'
    var_6 = bool(var_4 == var_5 and var_3.remove_redundant_aliases)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_239_evaluates_to_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'as'
    var_2 = 'alias'
    var_3 = [var_0, var_1, var_2]
    var_4 = var_1 in var_3
    var_5 = 1
    var_6 = len(var_3)



# Parsed testcases at query #15
#--------------------------




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
    var_0 = "print('hello')"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello\\'world')"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("""hello"""'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

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
    var_0 = "import sys; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "from sys import path; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '### comment'
    var_1 = ''
    var_2 = 0
    var_3 = '###'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "import sys; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("hello"); print(\'world\')'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("""hello'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'from'
    var_1 = 'module'
    var_2 = 'existing_alias'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'module'
    var_7 = 'existing_alias'
    var_8 = bool(not var_7 not in var_5['from'][var_6])
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_241_evaluates_to_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'as'
    var_2 = 'alias'
    var_3 = [var_0, var_1, var_2]
    var_4 = var_1 in var_3
    var_5 = 1
    var_6 = len(var_3)



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = 'some_comment'
    var_2 = [var_1]
    var_3 = bool(not (var_0 and var_2 is not None))
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 9/12 statements.
# Partially parsed test_file_contents_from_import. Retrieved 10/16 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 9/12 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 11/17 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 9/12 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'os'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 1
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_27 is True
    var_28 = var_1.verbose_output
    var_29 = bool(var_1.verbose_output == [])
    assert var_29 is True
    var_30 = set()
    var_31 = var_1.trailing_commas
    var_32 = bool(var_1.trailing_commas == var_30)
    assert var_32 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': []}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -1
    var_25 = var_1.original_line_count
    assert var_25 == 1
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_28 is True
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os  # comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'os'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 1
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_27 is True
    var_28 = var_1.verbose_output
    var_29 = bool(var_1.verbose_output == [])
    assert var_29 is True
    var_30 = set()
    var_31 = var_1.trailing_commas
    var_32 = bool(var_1.trailing_commas == var_30)
    assert var_32 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    sys\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    sys', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': [], 'os.sys': []}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = 'sys'
    var_20 = (var_19, var_17)
    var_21 = [var_18, var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -4
    var_27 = var_1.original_line_count
    assert var_27 == 4
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == {'os'})
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort:imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'numpy'
    var_16 = True
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = [var_18]
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -2
    var_24 = var_1.original_line_count
    assert var_24 == 2
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_27 is True
    var_28 = var_1.verbose_output
    var_29 = bool(var_1.verbose_output == [])
    assert var_29 is True
    var_30 = set()
    var_31 = var_1.trailing_commas
    var_32 = bool(var_1.trailing_commas == var_30)
    assert var_32 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:skip\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:skip', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort:skip', 'import os'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 2
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_content. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 13/20 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == [''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == -1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = set()
    var_25 = var_1.trailing_commas
    var_26 = bool(var_1.trailing_commas == var_24)
    assert var_26 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -1
    var_27 = var_1.original_line_count
    assert var_27 == 1
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_30 is True
    var_31 = var_1.verbose_output
    var_32 = bool(var_1.verbose_output == [])
    assert var_32 is True
    var_33 = set()
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == var_33)
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nx = 1\nfrom sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'x = 1', 'from sys import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['x = 1'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = 'sys'
    var_22 = 'path'
    var_23 = (var_22, var_17)
    var_24 = [var_23]
    var_25 = [var_24]
    var_26 = var_1.imports
    var_27 = var_1.categorized_comments
    var_28 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_28 is True
    var_29 = var_1.change_count
    assert var_29 == -2
    var_30 = var_1.original_line_count
    assert var_30 == 3
    var_31 = var_1.line_separator
    assert var_31 == '\n'
    var_32 = var_1.sections
    var_33 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_33 is True
    var_34 = var_1.verbose_output
    var_35 = bool(var_1.verbose_output == [])
    assert var_35 is True
    var_36 = set()
    var_37 = var_1.trailing_commas
    var_38 = bool(var_1.trailing_commas == var_36)
    assert var_38 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# Comment\nimport os # inline comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# Comment', 'import os # inline comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# Comment'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'os'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' inline comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 1
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    environ\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    environ', ')'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = True
    var_20 = (var_18, var_19)
    var_21 = 'environ'
    var_22 = (var_21, var_19)
    var_23 = [var_20, var_22]
    var_24 = [var_23]
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == -4
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_32 is True
    var_33 = var_1.verbose_output
    var_34 = bool(var_1.verbose_output == [])
    assert var_34 is True
    var_35 = set()
    var_36 = var_1.trailing_commas
    var_37 = bool(var_1.trailing_commas == var_35)
    assert var_37 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: imports-thirdparty', 'import numpy'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort: imports-thirdparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort: imports-thirdparty': 'THIRDPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_24 is True
    var_25 = var_1.change_count
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_29 is True
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_new_comment_appended_to_comments_list. Retrieved 2/3 statements.


def test_case_0():
    var_0 = []
    var_1 = 'This is a comment'
    var_2 = bool(var_0 == ['This is a comment'])
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_non_import_line. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 10/17 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/19 statements.


import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [''])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == -1
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
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
    assert var_16 == 1
    var_17 = var_1.line_separator
    assert var_17 == '\n'
    var_18 = var_1.sections
    var_19 = var_1.verbose_output
    var_20 = bool(var_1.verbose_output == [])
    assert var_20 is True
    var_21 = set()
    var_22 = var_1.trailing_commas
    var_23 = bool(var_1.trailing_commas == var_21)
    assert var_23 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = True
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = []
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 1
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = []
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -1
    var_25 = var_1.original_line_count
    assert var_25 == 1
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = var_1.verbose_output
    var_29 = bool(var_1.verbose_output == [])
    assert var_29 is True
    var_30 = set()
    var_31 = var_1.trailing_commas
    var_32 = bool(var_1.trailing_commas == var_30)
    assert var_32 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = True
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = []
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 1
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {'os': ['operating_system']}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os as operating_system'
    var_15 = True
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = []
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 1
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == ['x = 1'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 1
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = True
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = []
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == 0
    var_24 = var_1.original_line_count
    assert var_24 == 2
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports-thirdparty\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {'THIRDPARTY': []})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {'# isort: imports-thirdparty': 'THIRDPARTY'})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = 'os'
    var_15 = True
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = []
    var_20 = var_1.imports
    var_21 = var_1.categorized_comments
    var_22 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_22 is True
    var_23 = var_1.change_count
    assert var_23 == -1
    var_24 = var_1.original_line_count
    assert var_24 == 2
    var_25 = var_1.line_separator
    assert var_25 == '\n'
    var_26 = var_1.sections
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = []
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = var_1.imports
    var_22 = var_1.categorized_comments
    var_23 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_23 is True
    var_24 = var_1.change_count
    assert var_24 == -3
    var_25 = var_1.original_line_count
    assert var_25 == 3
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = var_1.verbose_output
    var_29 = bool(var_1.verbose_output == [])
    assert var_29 is True
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == {'os'})
    assert var_31 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: skip\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == ['# isort: skip', 'import os'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == -1
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
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
    assert var_16 == 2
    var_17 = var_1.line_separator
    assert var_17 == '\n'
    var_18 = var_1.sections
    var_19 = var_1.verbose_output
    var_20 = bool(var_1.verbose_output == [])
    assert var_20 is True
    var_21 = set()
    var_22 = var_1.trailing_commas
    var_23 = bool(var_1.trailing_commas == var_21)
    assert var_23 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import (\n    path,\n    join\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.lines_without_imports
    var_3 = bool(var_1.lines_without_imports == [])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.place_imports
    var_6 = bool(var_1.place_imports == {})
    assert var_6 is True
    var_7 = var_1.import_placements
    var_8 = bool(var_1.import_placements == {})
    assert var_8 is True
    var_9 = var_1.as_map
    var_10 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_10 is True
    var_11 = 'THIRDPARTY'
    var_12 = 'straight'
    var_13 = 'from'
    var_14 = []
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = 'join'
    var_20 = (var_19, var_17)
    var_21 = [var_18, var_20]
    var_22 = [var_21]
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True



