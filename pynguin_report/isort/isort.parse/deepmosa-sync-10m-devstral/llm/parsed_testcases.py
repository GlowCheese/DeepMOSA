####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['straight']['os']
    assert var_2 is True
    var_3 = var_1.imports['THIRDPARTY']['straight']['sys']
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from collections import defaultdict\nfrom typing import List\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'collections'
    var_3 = bool('collections' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'defaultdict'
    var_5 = bool('defaultdict' in var_1.imports['STDLIB']['from']['collections'])
    assert var_5 is True
    var_6 = 'typing'
    var_7 = bool('typing' in var_1.imports['TYPING']['from'])
    assert var_7 is True
    var_8 = 'List'
    var_9 = bool('List' in var_1.imports['TYPING']['from']['typing'])
    assert var_9 is True
    var_10 = var_1.lines_without_imports
    var_11 = bool(var_1.lines_without_imports == [])
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1\nimport os\n\ndef foo():\n    pass\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['straight']['os']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['x = 1', '', '\ndef foo():\n    pass\n'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os  # inline comment\n# Another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['straight']['os']
    assert var_2 is True
    var_3 = '# This is a comment'
    var_4 = bool('# This is a comment' in var_1.lines_without_imports)
    assert var_4 is True
    var_5 = '# Another comment'
    var_6 = bool('# Another comment' in var_1.lines_without_imports)
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import numpy as np\nfrom pandas import DataFrame as DF\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.as_map['straight']['numpy']
    var_3 = bool(var_1.as_map['straight']['numpy'] == ['np'])
    assert var_3 is True
    var_4 = var_1.as_map['from']['pandas.DataFrame']
    var_5 = bool(var_1.as_map['from']['pandas.DataFrame'] == ['DF'])
    assert var_5 is True
    var_6 = var_1.imports['THIRDPARTY']['straight']['numpy']
    assert var_6 is False
    var_7 = 'DataFrame'
    var_8 = bool('DataFrame' in var_1.imports['THIRDPARTY']['from']['pandas'])
    assert var_8 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from typing import (\n    List,\n    Dict,\n    Optional,\n)\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'typing'
    var_3 = bool('typing' in var_1.imports['TYPING']['from'])
    assert var_3 is True
    var_4 = 'List'
    var_5 = bool('List' in var_1.imports['TYPING']['from']['typing'])
    assert var_5 is True
    var_6 = 'Dict'
    var_7 = bool('Dict' in var_1.imports['TYPING']['from']['typing'])
    assert var_7 is True
    var_8 = 'Optional'
    var_9 = bool('Optional' in var_1.imports['TYPING']['from']['typing'])
    assert var_9 is True
    var_10 = 'typing'
    var_11 = bool('typing' in var_1.trailing_commas)
    assert var_11 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports\nimport os\n# isort: imports-end\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == 0
    var_3 = var_1.imports['THIRDPARTY']['straight']['os']
    assert var_3 is True
    var_4 = var_1.imports['THIRDPARTY']['straight']['sys']
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = bool(var_1.imports == {})
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# Just a comment\n# Another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports
    var_3 = bool(var_1.imports == {})
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# Just a comment', '# Another comment'])
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = "import os"\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['straight']['sys']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['x = "import os"'])
    assert var_4 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""import os"""\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['THIRDPARTY']['straight']['sys']
    assert var_2 is True
    var_3 = var_1.lines_without_imports
    var_4 = bool(var_1.lines_without_imports == ['"""import os"""'])
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # noqa'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'from os import path  # NOQA'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'from os import path  # isort: skip'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None
    var_4 = 'import os  # isort: split'
    var_5 = module_0.import_type(var_4)
    assert var_5 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'def foo(): pass'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = '\n'
    var_2 = 'line_ending'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.line_separator
    assert var_6 == '\n'



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import nested_module as alias\n'
    var_1 = True
    var_2 = 'remove_redundant_aliases'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_file_contents_single_line_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 12/16 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_from_as. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 12/19 statements.


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
    var_0 = 'from os import path as os_path'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path as os_path'])
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
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': ['os_path']}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'os'
    var_18 = 'path'
    var_19 = False
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as. Retrieved 10/14 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 13/20 statements.
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
    assert var_17 == -1
    var_18 = var_1.original_line_count
    assert var_18 == 1
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = var_1.verbose_output
    var_22 = bool(var_1.verbose_output == [])
    assert var_22 is True
    var_23 = set()
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == var_23)
    assert var_25 is True

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
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

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
    var_14 = 'from'
    var_15 = 'straight'
    var_16 = 'sys'
    var_17 = 'path'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = []
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
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

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
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

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
    var_14 = 'from'
    var_15 = 'straight'
    var_16 = 'os'
    var_17 = 'path'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = 'environ'
    var_21 = (var_20, var_18)
    var_22 = [var_19, var_21]
    var_23 = [var_22]
    var_24 = []
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == -1
    var_29 = var_1.original_line_count
    assert var_29 == 4
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = var_1.verbose_output
    var_33 = bool(var_1.verbose_output == [])
    assert var_33 is True
    var_34 = set()
    var_35 = var_1.trailing_commas
    var_36 = bool(var_1.trailing_commas == var_34)
    assert var_36 is True

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
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path,'])
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
    var_14 = 'from'
    var_15 = 'straight'
    var_16 = 'os'
    var_17 = 'path'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = []
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
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == {'os'})
    assert var_33 is True

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
    var_7 = bool(var_5.verbose_output == ['else-type place_module for os returned THIRDPARTY'])
    assert var_7 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: skip\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: skip', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort: skip', 'import os'])
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



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 'from module import ('
    var_1 = 0
    var_2 = 1
    var_3 = '('
    var_4 = 0
    var_5 = '#'
    var_6 = 1
    var_7 = var_0.split(var_5, var_6)[var_4]
    var_8 = var_3 in var_7
    var_9 = bool(var_8 and var_1 < var_2)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from unknown_module import something'
    var_3 = module_1.file_contents(var_2, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_placed_module_in_imports. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'os'
    var_1 = 'os'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = []
    var_5 = []



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 'module'
    var_1 = 'as'
    var_2 = 'alias'
    var_3 = 'another_module'
    var_4 = 'another_alias'
    var_5 = [var_0, var_1, var_2, var_3, var_1, var_4]
    var_6 = 'as'
    var_7 = bool('as' in var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nx = 1'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = 'import os'
    var_2 = -1
    var_3 = var_0[var_2]
    var_4 = ','
    var_5 = -1
    var_6 = -1
    var_7 = var_0[var_6]
    var_8 = var_1.split(var_7)[var_5]
    var_9 = var_4 in var_8
    var_10 = var_0 and var_3 and var_9
    var_11 = bool(not var_10)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 10/19 statements.


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
    var_9 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_content. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_aliases. Retrieved 14/21 statements.


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
    var_0 = "print('hello')\nimport sys\nfrom os import path"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ["print('hello')", 'import sys', 'from os import path'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ["print('hello')"])
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
    var_21 = 'os'
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
    var_0 = '# This is a comment\nimport json  # inline comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import json  # inline comment'])
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
    var_16 = 'json'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = []
    var_22 = var_1.imports
    var_23 = var_1.categorized_comments
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'json': [' inline comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
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
    var_0 = 'import numpy as np\nfrom pandas import DataFrame as DF'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np', 'from pandas import DataFrame as DF'])
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
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {'pandas.DataFrame': ['DF']}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = 'pandas'
    var_22 = 'DataFrame'
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_75_evaluates_to_false. Retrieved 6/11 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = "print('Hello, World!')"
    var_1 = True
    var_2 = 'float_to_top'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = 'import'
    var_7 = 'from'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_line_ends_with_backslash. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'import something \\'
    var_1 = '\\'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 4/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa'
    var_5 = 'noqa'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiple_imports. Retrieved 12/16 statements.
# Partially parsed test_file_contents_with_mixed_imports. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_code. Retrieved 10/14 statements.


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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'sys'
    var_18 = 'argv'
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
    assert var_27 == -2
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
    var_0 = 'import os\nfrom sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'from sys import argv'])
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
    var_22 = 'argv'
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



# Parsed testcases at query #19
#--------------------------




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
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {'os': True}, 'from': {}}})
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'from': {'sys': {'argv': True}}, 'straight': {}}})
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
    var_0 = 'import os\nx = 1\nfrom sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', 'x = 1', 'from sys import argv'])
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
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {'os': True}, 'from': {'sys': {'argv': True}}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == -2
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

import isort.parse as module_0

def test_case_0():
    var_0 = '# This is a comment\nimport os # inline comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os # inline comment'])
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
    assert var_17 == -1
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
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'straight': {'numpy as np': True}, 'from': {}}})
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
    var_0 = 'from numpy import array as arr'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from numpy import array as arr'])
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
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'numpy.array': ['arr']}})
    assert var_12 is True
    var_13 = var_1.imports
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'from': {'numpy': {'array as arr': True}}, 'straight': {}}})
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
    var_0 = 'from numpy import (\n    array,\n    matrix,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from numpy import (', '    array,', '    matrix,', ')'])
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
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'from': {'numpy': {'array': True, 'matrix': True}}, 'straight': {}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == -4
    var_18 = var_1.original_line_count
    assert var_18 == 4
    var_19 = var_1.line_separator
    assert var_19 == '\n'
    var_20 = var_1.sections
    var_21 = bool(var_1.sections == ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'])
    assert var_21 is True
    var_22 = var_1.verbose_output
    var_23 = bool(var_1.verbose_output == [])
    assert var_23 is True
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == {'numpy'})
    assert var_25 is True

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
    var_9 = bool(var_1)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiple_imports. Retrieved 12/16 statements.


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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'sys'
    var_18 = 'argv'
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
    var_0 = 'import os  # Comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os  # Comment'])
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
    var_24 = bool(var_1.categorized_comments == {'from': {}, 'straight': {'os': [' Comment']}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
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
    var_0 = 'import os, sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os, sys'])
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



# Parsed testcases at query #21
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\r\nline2\r\nline3'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\rline2\rline3'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'

import isort.parse as module_0

def test_case_0():
    var_0 = 'line1\r\nline2\rline3'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\r\n'

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'single line'
    var_1 = module_0._infer_line_separator(var_0)
    assert var_1 == '\n'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_code_and_imports. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 10/14 statements.
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = []
    var_17 = 'sys'
    var_18 = 'argv'
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
    var_0 = "print('hello')\nimport os\nx = 1"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ["print('hello')", 'import os', 'x = 1'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ["print('hello')", 'x = 1'])
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
    assert var_26 == 3
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
    var_0 = '# This is a comment\nimport os # inline comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os # inline comment'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [])
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_82_evaluates_to_true. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'import foo  # isort:skip ('
    var_1 = 0
    var_2 = '#'
    var_3 = 1
    var_4 = var_0.split(var_2, var_3)[var_1]



# Parsed testcases at query #24
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from module import name  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #25
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
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'print("""hello"""world)'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
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
    var_0 = "import os; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "from os import path; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = '#'
    var_4 = (var_3,)
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello'); print('world')"
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
    var_0 = "print('hello\\\\world')"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "cimport numpy; print('hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True



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
    var_0 = "'test"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"test'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'''test"
    var_1 = "'''"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'''"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"""test'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"""'))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'test\\'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
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
    var_0 = '# comment'
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
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os; import sys'
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
    var_0 = 'import os; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, ''))
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
    var_0 = "# 'quote"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (False, ''))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'''multiline"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'''"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "'test\\\\'"
    var_1 = "'"
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, "'"))
    assert var_5 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '"test\''
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = module_0.skip_line(var_0, var_1, var_2, var_3)
    var_5 = bool(var_4 == (True, '"'))
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.parse as module_0

def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'

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

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'from os import path  # isort:split'
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
    var_4 = 'import os  # NOQA'
    var_5 = module_1.import_type(var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa   '
    var_5 = module_1.import_type(var_4)
    assert var_5 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort: skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'import os  #isort:skip'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None



# Parsed testcases at query #3
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'line = "test"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = var_5[0]
    assert var_6 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_file_contents_empty_string. Retrieved 5/8 statements.
# Partially parsed test_file_contents_single_import. Retrieved 10/17 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/21 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/17 statements.
# Partially parsed test_file_contents_with_alias. Retrieved 14/20 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 12/22 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/17 statements.


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
    assert var_18 == -1
    var_19 = var_1.original_line_count
    assert var_19 == 1
    var_20 = var_1.line_separator
    assert var_20 == '\n'
    var_21 = var_1.sections
    var_22 = bool(var_1.sections == [])
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
    var_14 = 'THIRDPARTY'
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
    var_28 = bool(var_1.sections == [])
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
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'THIRDPARTY'
    var_15 = []
    var_16 = 'os'
    var_17 = 'path'
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
    var_14 = 'THIRDPARTY'
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
    var_28 = bool(var_1.sections == [])
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
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = 'os'
    var_14 = 'operating_system'
    var_15 = [var_14]
    var_16 = {var_13: var_15}
    var_17 = var_1.as_map
    var_18 = 'THIRDPARTY'
    var_19 = 'os as operating_system'
    var_20 = False
    var_21 = (var_19, var_20)
    var_22 = [var_21]
    var_23 = [var_22]
    var_24 = []
    var_25 = var_1.imports
    var_26 = var_1.categorized_comments
    var_27 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_27 is True
    var_28 = var_1.change_count
    assert var_28 == -1
    var_29 = var_1.original_line_count
    assert var_29 == 1
    var_30 = var_1.line_separator
    assert var_30 == '\n'
    var_31 = var_1.sections
    var_32 = bool(var_1.sections == [])
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
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'THIRDPARTY'
    var_15 = []
    var_16 = 'os'
    var_17 = 'path'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = 'sys'
    var_21 = (var_20, var_18)
    var_22 = [var_19, var_21]
    var_23 = [var_22]
    var_24 = var_1.imports
    var_25 = var_1.categorized_comments
    var_26 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_26 is True
    var_27 = var_1.change_count
    assert var_27 == -1
    var_28 = var_1.original_line_count
    assert var_28 == 4
    var_29 = var_1.line_separator
    assert var_29 == '\n'
    var_30 = var_1.sections
    var_31 = bool(var_1.sections == [])
    assert var_31 is True
    var_32 = var_1.verbose_output
    var_33 = bool(var_1.verbose_output == [])
    assert var_33 is True
    var_34 = var_1.trailing_commas
    var_35 = bool(var_1.trailing_commas == {'os'})
    assert var_35 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort:imports-thirdparty\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import os'])
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
    var_11 = 'straight'
    var_12 = 'from'
    var_13 = var_1.as_map
    var_14 = 'THIRDPARTY'
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
    assert var_25 == 2
    var_26 = var_1.line_separator
    assert var_26 == '\n'
    var_27 = var_1.sections
    var_28 = bool(var_1.sections == [])
    assert var_28 is True
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = bool(not (var_0 and var_0 not in var_1))
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_428_evaluates_to_false. Retrieved 12/15 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 428 evaluates to False.'
    var_1 = 'import os'
    var_2 = 'SECTION1'
    var_3 = [var_2]
    var_4 = []
    var_5 = 'sections'
    var_6 = 'forced_separate'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1.file_contents(var_1, var_8)
    var_10 = {}
    var_11 = 'straight'
    var_12 = {}
    var_13 = 'os'
    var_14 = False



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = 'import os, sys'
    var_2 = -1
    var_3 = var_0[var_2]
    var_4 = ','
    var_5 = -1
    var_6 = -1
    var_7 = var_0[var_6]
    var_8 = var_1.split(var_7)[var_5]
    var_9 = var_4 in var_8
    var_10 = var_0 and var_3 and var_9
    assert var_10 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_56_evaluates_to_true. Retrieved 2/3 statements.


def test_case_0():
    var_0 = '# isort: imports-thirdparty'
    var_1 = '#'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_142_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'from'
    var_1 = ' '
    var_2 = ' as '
    var_3 = ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_226_evaluates_to_true. Retrieved 15/23 statements.


def test_case_0():
    var_0 = True
    var_1 = 'from module'
    var_2 = 'item1'
    var_3 = 'item2'
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = ' '
    var_8 = ' cimport '
    var_9 = ' import '
    var_10 = var_8 if var_0 else var_9
    var_11 = ''
    var_12 = 1
    var_13 = var_4[var_12:]
    var_14 = *var_13



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = '# comment'
    var_1 = [var_0]
    var_2 = []
    var_3 = bool(var_1 and var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from os import path'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_428_evaluates_to_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = ''
    var_1 = 'section'
    var_2 = 'straight'
    var_3 = 'from'
    var_4 = []
    var_5 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_file_contents_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = 'from module cimport something'
    var_1 = ' cimport '
    var_2 = bool(' cimport ' in var_0)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 9/12 statements.
# Partially parsed test_file_contents_from_import. Retrieved 10/16 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 9/12 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 11/17 statements.
# Partially parsed test_file_contents_with_as. Retrieved 9/12 statements.
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
    var_21 = var_1.verbose_output
    var_22 = bool(var_1.verbose_output == [])
    assert var_22 is True
    var_23 = set()
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == var_23)
    assert var_25 is True

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
    var_27 = var_1.verbose_output
    var_28 = bool(var_1.verbose_output == [])
    assert var_28 is True
    var_29 = set()
    var_30 = var_1.trailing_commas
    var_31 = bool(var_1.trailing_commas == var_29)
    assert var_31 is True

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
    var_14 = 'from'
    var_15 = 'sys'
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
    var_0 = 'from os import (\n    path,\n    sep\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    sep', ')'])
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
    var_14 = 'from'
    var_15 = 'os'
    var_16 = 'path'
    var_17 = True
    var_18 = (var_16, var_17)
    var_19 = 'sep'
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
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == {'os'})
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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'numpy'
    var_16 = False
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
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: imports-thirdparty', 'import os'])
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_ends_with_backslash. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'import something \\'
    var_1 = '\\'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = bool(not (var_0 and var_1 is None))
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
    var_2 = 'cimport numpy'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'

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
    var_6 = 'from os import path  # noqa'
    var_7 = 'honor_noqa'
    var_8 = {var_7: var_1}
    var_9 = module_0.Config(**var_8)
    var_10 = module_1.import_type(var_6, var_9)
    assert var_10 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'from os import path  # isort: skip'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None
    var_4 = 'import os  # isort: split'
    var_5 = module_0.import_type(var_4)
    assert var_5 is None

import isort.parse as module_0

def test_case_0():
    var_0 = 'x = 1'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None
    var_2 = 'def foo():'
    var_3 = module_0.import_type(var_2)
    assert var_3 is None
    var_4 = 'class Bar:'
    var_5 = module_0.import_type(var_4)
    assert var_5 is None

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
    var_6 = 'from os import path  # NoQa'
    var_7 = 'honor_noqa'
    var_8 = {var_7: var_1}
    var_9 = module_0.Config(**var_8)
    var_10 = module_1.import_type(var_6, var_9)
    assert var_10 is None

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
    var_6 = 'from os import path  # noqa'
    var_7 = 'honor_noqa'
    var_8 = {var_7: var_1}
    var_9 = module_0.Config(**var_8)
    var_10 = module_1.import_type(var_6, var_9)
    assert var_10 == 'from'



# Parsed testcases at query #21
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nx = 1'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------




import isort.parse as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['', ''])
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
    var_3 = bool(var_1.in_lines == ['import os', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_0 = 'from sys import argv'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import argv', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {}, 'from': {'sys': {'argv': True}}}})
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
    var_0 = '# This is a comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# This is a comment', ''])
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
    var_0 = 'from os import (\n    path,\n    environ\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import (', '    path,', '    environ', ')', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {}, 'from': {'os': {'path': True, 'environ': True}}}})
    assert var_14 is True
    var_15 = var_1.categorized_comments
    var_16 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_16 is True
    var_17 = var_1.change_count
    assert var_17 == 0
    var_18 = var_1.original_line_count
    assert var_18 == 4
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
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_14 = bool(var_1.imports == {'THIRDPARTY': {'straight': {'numpy': False}, 'from': {}}})
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
    var_0 = '# isort:imports-thirdparty\nimport numpy'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort:imports-thirdparty', 'import numpy', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_0 = 'from os import path, environ,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path, environ,', ''])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == [''])
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
    var_14 = bool(var_1.imports == {'STDLIB': {'straight': {}, 'from': {'os': {'path': True, 'environ': True}}}})
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
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == {'os'})
    assert var_25 is True

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
    var_7 = bool(var_5.verbose_output == ['else-type place_module for os returned STDLIB'])
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_397_evaluates_to_true. Retrieved 6/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '# isort: imports-'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = '#'
    var_4 = '"""'
    var_5 = "'''"
    var_6 = var_2.treat_comments_as_code



# Parsed testcases at query #24
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\nimport sys\n\n# Comment\nx = 1\n'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.import_index
    var_5 = bool(var_3.import_index != 3)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_273. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'from module import nested_module as as_name  # comment'
    var_1 = True
    var_2 = 'combine_as_imports'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = 'from'
    var_7 = var_5.categorized_comments[var_6]
    var_8 = 'module.__combined_as__'



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'import os\n\nx = 1\n'
    var_1 = True
    var_2 = 'float_to_top'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.import_index
    assert var_6 == 1



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = 'import os, sys'
    var_2 = -1
    var_3 = var_0[var_2]
    var_4 = ','
    var_5 = -1
    var_6 = -1
    var_7 = var_0[var_6]
    var_8 = var_1.split(var_7)[var_5]
    var_9 = var_4 in var_8
    var_10 = var_0 and var_3 and var_9
    var_11 = bool(not var_10)
    assert var_11 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_content. Retrieved 15/25 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_trailing_comma. Retrieved 10/17 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
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
    var_21 = var_1.verbose_output
    var_22 = bool(var_1.verbose_output == [])
    assert var_22 is True
    var_23 = set()
    var_24 = var_1.trailing_commas
    var_25 = bool(var_1.trailing_commas == var_23)
    assert var_25 is True

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
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STDLIB'
    var_14 = 'from'
    var_15 = 'straight'
    var_16 = 'sys'
    var_17 = 'argv'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = []
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
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = set()
    var_33 = var_1.trailing_commas
    var_34 = bool(var_1.trailing_commas == var_32)
    assert var_34 is True

import isort.parse as module_0

def test_case_0():
    var_0 = "print('hello')\nimport os\nfrom sys import argv"
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ["print('hello')", 'import os', 'from sys import argv'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ["print('hello')"])
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
    var_14 = 'STDLIB'
    var_15 = 'straight'
    var_16 = 'from'
    var_17 = 'os'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = []
    var_23 = 'sys'
    var_24 = 'argv'
    var_25 = (var_24, var_18)
    var_26 = [var_25]
    var_27 = [var_26]
    var_28 = []
    var_29 = var_1.imports
    var_30 = var_1.categorized_comments
    var_31 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_31 is True
    var_32 = var_1.change_count
    assert var_32 == -2
    var_33 = var_1.original_line_count
    assert var_33 == 3
    var_34 = var_1.line_separator
    assert var_34 == '\n'
    var_35 = var_1.sections
    var_36 = var_1.verbose_output
    var_37 = bool(var_1.verbose_output == [])
    assert var_37 is True
    var_38 = set()
    var_39 = var_1.trailing_commas
    var_40 = bool(var_1.trailing_commas == var_38)
    assert var_40 is True

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
    assert var_25 == -1
    var_26 = var_1.original_line_count
    assert var_26 == 2
    var_27 = var_1.line_separator
    assert var_27 == '\n'
    var_28 = var_1.sections
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = 'from sys import (\n    argv,\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from sys import (', '    argv,', ')'])
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
    var_14 = 'from'
    var_15 = 'straight'
    var_16 = 'sys'
    var_17 = 'argv'
    var_18 = True
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = [var_20]
    var_22 = []
    var_23 = var_1.imports
    var_24 = var_1.categorized_comments
    var_25 = bool(var_1.categorized_comments == {'from': {}, 'straight': {}, 'nested': {}, 'above': {'straight': {}, 'from': {}}})
    assert var_25 is True
    var_26 = var_1.change_count
    assert var_26 == -2
    var_27 = var_1.original_line_count
    assert var_27 == 3
    var_28 = var_1.line_separator
    assert var_28 == '\n'
    var_29 = var_1.sections
    var_30 = var_1.verbose_output
    var_31 = bool(var_1.verbose_output == [])
    assert var_31 is True
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == {'sys'})
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
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True

import isort.parse as module_0

def test_case_0():
    var_0 = '# isort: imports-firstparty\nimport my_module'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: imports-firstparty', 'import my_module'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort: imports-firstparty'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 1
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'FIRSTPARTY': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort: imports-firstparty': 'FIRSTPARTY'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'FIRSTPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'my_module'
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
    var_29 = var_1.verbose_output
    var_30 = bool(var_1.verbose_output == [])
    assert var_30 is True
    var_31 = set()
    var_32 = var_1.trailing_commas
    var_33 = bool(var_1.trailing_commas == var_31)
    assert var_33 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_320_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_mixed_content. Retrieved 14/21 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_aliases. Retrieved 14/21 statements.


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
    var_0 = 'import os\n\nx = 1\nfrom collections import defaultdict'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import os', '', 'x = 1', 'from collections import defaultdict'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['', 'x = 1'])
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
    var_21 = 'collections'
    var_22 = 'defaultdict'
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
    assert var_30 == 4
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
    var_0 = '# This is a comment\nimport os  # inline comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# This is a comment', 'import os  # inline comment'])
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
    var_0 = 'import numpy as np\nfrom pandas import DataFrame as DF'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['import numpy as np', 'from pandas import DataFrame as DF'])
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
    var_12 = bool(var_1.as_map == {'straight': {'numpy': ['np']}, 'from': {'pandas.DataFrame': ['DF']}})
    assert var_12 is True
    var_13 = 'THIRDPARTY'
    var_14 = 'straight'
    var_15 = 'from'
    var_16 = 'numpy'
    var_17 = False
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_19]
    var_21 = 'pandas'
    var_22 = 'DataFrame'
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_file_contents_single_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/18 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 12/16 statements.
# Partially parsed test_file_contents_with_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_section_comment. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_as_import. Retrieved 10/14 statements.
# Partially parsed test_file_contents_with_multiline_import. Retrieved 13/20 statements.


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
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': ['path']}})
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
    assert var_27 == -2
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
    var_0 = '# isort: imports-standard\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['# isort: imports-standard', 'import os'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports
    var_5 = bool(var_1.lines_without_imports == ['# isort: imports-standard'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0
    var_7 = var_1.place_imports
    var_8 = bool(var_1.place_imports == {'STANDARD': []})
    assert var_8 is True
    var_9 = var_1.import_placements
    var_10 = bool(var_1.import_placements == {'# isort: imports-standard': 'STANDARD'})
    assert var_10 is True
    var_11 = var_1.as_map
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {}})
    assert var_12 is True
    var_13 = 'STANDARD'
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
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': ['path'], 'os.sys': ['sys']}})
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
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.in_lines
    var_3 = bool(var_1.in_lines == ['from os import path,'])
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
    var_12 = bool(var_1.as_map == {'straight': {}, 'from': {'os.path': ['path']}})
    assert var_12 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_428_evaluates_to_false. Retrieved 10/11 statements.


import isort.settings as module_0
import isort.parse as module_1

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'sections'
    var_4 = 'forced_separate'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = 'import module1\nimport module2'
    var_8 = module_1.file_contents(var_7, var_6)
    var_9 = 'straight'
    var_10 = var_8.imports[var_0][var_9]
    var_11 = 'module1'
    var_12 = False



# Parsed testcases at query #33
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
    var_8 = 0
    var_9 = 1
    var_10 = 'module'
    var_11 = False
    var_12 = []
    var_13 = 'treat_all_comments_as_code'
    var_14 = 'treat_comments_as_code'
    var_15 = {var_13: var_11, var_14: var_12}
    var_16 = module_0.Config(**var_15)
    var_17 = bool(var_2)
    assert var_17 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_241. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'as'
    var_2 = 'alias'
    var_3 = 'another'
    var_4 = 'another_alias'
    var_5 = [var_0, var_1, var_2, var_3, var_1, var_4]
    var_6 = 1
    var_7 = len(var_5)



# Parsed testcases at query #35
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'straight'
    var_1 = False
    var_2 = 'remove_redundant_aliases'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'module'
    var_6 = 'module'
    var_7 = 'from'
    var_8 = var_0 == var_7
    var_9 = var_4.remove_redundant_aliases
    var_10 = -1
    var_11 = '.'
    var_12 = var_6.split(var_11)[var_10]
    var_13 = var_5 == var_12
    var_14 = var_9 and var_13
    var_15 = var_8 or var_14
    var_16 = bool(not var_15)
    assert var_16 is True



