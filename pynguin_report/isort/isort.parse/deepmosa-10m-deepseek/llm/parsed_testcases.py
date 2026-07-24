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
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = "print('Hello')"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = '"""Hello"""'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, '"""'))
    assert var_6 is True


def test_case_0():
    var_0 = "'''Hello'''"
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, "'''"))
    assert var_6 is True


def test_case_0():
    var_0 = 'print("He said \\"Hi\\"")'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'print("Hello")  # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'x = 1; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'import sys; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'from os import path; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'cimport numpy; x = 1'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'x = 1; y = 2  # comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'x = 1; y = 2'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = False
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'print("Hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'print("Hello")'
    var_1 = '"""'
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, '"""'))
    assert var_6 is True


def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = '# comment'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = '# "comment"'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'import sys'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'from os import path'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'import sys; x = 1; from os import path'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (False, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'x = 1; import sys'
    var_1 = ''
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True


def test_case_0():
    var_0 = 'print("Hello")'
    var_1 = '"'
    var_2 = 0
    var_3 = ()
    var_4 = True
    var_5 = module_0.skip_line(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 == (True, ''))
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'


def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'


def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'


def test_case_0():
    var_0 = "print('hello')"
    var_1 = module_0.import_type(var_0)
    assert var_1 is None

import isort.parse as module_1
import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None


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


def test_case_0():
    var_0 = 'import os  # isort: skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = 'import os  # isort: split'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = ''
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = 'fromage cheese'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = '  import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'


def test_case_0():
    var_0 = 'from sys import path   '
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0


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
    var_9 = 'import os\nimport sys\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 0
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['straight'])
    assert var_13 is True
    var_14 = 'sys'
    var_15 = bool('sys' in var_10.imports['STDLIB']['straight'])
    assert var_15 is True
    var_16 = var_10.lines_without_imports
    var_17 = bool(var_10.lines_without_imports == [])
    assert var_17 is True


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
    var_9 = 'from os import path\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 0
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['from'])
    assert var_13 is True
    var_14 = 'path'
    var_15 = bool('path' in var_10.imports['STDLIB']['from']['os'])
    assert var_15 is True
    var_16 = var_10.lines_without_imports
    var_17 = bool(var_10.lines_without_imports == [])
    assert var_17 is True


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
    var_9 = 'import os  # comment\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 0
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['straight'])
    assert var_13 is True
    var_14 = var_10.categorized_comments['straight']['os']
    var_15 = bool(var_10.categorized_comments['straight']['os'] == [' comment'])
    assert var_15 is True


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
    var_9 = 'from os import (\n    path,\n    sep\n)\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 0
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['from'])
    assert var_13 is True
    var_14 = 'path'
    var_15 = bool('path' in var_10.imports['STDLIB']['from']['os'])
    assert var_15 is True
    var_16 = 'sep'
    var_17 = bool('sep' in var_10.imports['STDLIB']['from']['os'])
    assert var_17 is True
    var_18 = var_10.lines_without_imports
    var_19 = bool(var_10.lines_without_imports == [])
    assert var_19 is True


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
    var_9 = 'import os as operating_system\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 0
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['straight'])
    assert var_13 is True
    var_14 = var_10.as_map['straight']['os']
    var_15 = bool(var_10.as_map['straight']['os'] == ['operating_system'])
    assert var_15 is True
    var_16 = var_10.lines_without_imports
    var_17 = bool(var_10.lines_without_imports == [])
    assert var_17 is True


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
    var_9 = "print('hello')\nimport os\nprint('world')\n"
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == 1
    var_12 = 'os'
    var_13 = bool('os' in var_10.imports['STDLIB']['straight'])
    assert var_13 is True
    var_14 = var_10.lines_without_imports
    var_15 = bool(var_10.lines_without_imports == ["print('hello')", "print('world')"])
    assert var_15 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = 'os'
    var_7 = [var_6]
    var_8 = 'sections'
    var_9 = 'forced_separate'
    var_10 = {var_8: var_5, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = 'import os\nimport sys\n'
    var_13 = module_1.file_contents(var_12, var_11)
    var_14 = 'os'
    var_15 = bool('os' in var_13.imports['os']['straight'])
    assert var_15 is True
    var_16 = 'sys'
    var_17 = bool('sys' in var_13.imports['STDLIB']['straight'])
    assert var_17 is True
    var_18 = var_13.lines_without_imports
    var_19 = bool(var_13.lines_without_imports == [])
    assert var_19 is True


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
    var_9 = 'from os import path,\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = 'os'
    var_12 = bool('os' in var_10.trailing_commas)
    assert var_12 is True
    var_13 = var_10.lines_without_imports
    var_14 = bool(var_10.lines_without_imports == [])
    assert var_14 is True


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
    var_9 = '# isort:imports-STDLIB\nimport os\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = 'STDLIB'
    var_12 = bool('STDLIB' in var_10.place_imports)
    assert var_12 is True
    var_13 = var_10.import_placements['# isort:imports-STDLIB']
    assert var_13 == 'STDLIB'
    var_14 = var_10.lines_without_imports
    var_15 = bool(var_10.lines_without_imports == [])
    assert var_15 is True


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
    var_9 = 'import os  # isort:skip\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == -1
    var_12 = var_10.lines_without_imports
    var_13 = bool(var_10.lines_without_imports == ['import os  # isort:skip'])
    assert var_13 is True


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
    var_9 = ''
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == -1
    var_12 = var_10.lines_without_imports
    var_13 = bool(var_10.lines_without_imports == [])
    assert var_13 is True
    var_14 = var_10.change_count
    assert var_14 == 0


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
    var_9 = '# comment\n'
    var_10 = module_1.file_contents(var_9, var_8)
    var_11 = var_10.import_index
    assert var_11 == -1
    var_12 = var_10.lines_without_imports
    var_13 = bool(var_10.lines_without_imports == ['# comment'])
    assert var_13 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = '# STDLIB'
    var_7 = [var_6]
    var_8 = 'sections'
    var_9 = 'section_comments'
    var_10 = {var_8: var_5, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = '# STDLIB\nimport os\n'
    var_13 = module_1.file_contents(var_12, var_11)
    var_14 = var_13.import_index
    assert var_14 == 0
    var_15 = 'os'
    var_16 = bool('os' in var_13.imports['STDLIB']['straight'])
    assert var_16 is True
    var_17 = var_13.lines_without_imports
    var_18 = bool(var_13.lines_without_imports == [])
    assert var_18 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = 'sections'
    var_8 = 'float_to_top'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = "print('start')\nimport os\n"
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = var_12.import_index
    assert var_13 == 0
    var_14 = 'os'
    var_15 = bool('os' in var_12.imports['STDLIB']['straight'])
    assert var_15 is True
    var_16 = var_12.lines_without_imports
    var_17 = bool(var_12.lines_without_imports == ["print('start')"])
    assert var_17 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = False
    var_8 = 'sections'
    var_9 = 'verbose'
    var_10 = 'only_modified'
    var_11 = {var_8: var_5, var_9: var_6, var_10: var_7}
    var_12 = module_0.Config(**var_11)
    var_13 = 'import os\n'
    var_14 = module_1.file_contents(var_13, var_12)
    var_15 = var_14.verbose_output
    var_16 = bool(var_14.verbose_output == [])
    assert var_16 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = 'sections'
    var_8 = 'remove_redundant_aliases'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = 'import os as os\n'
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = 'os'
    var_14 = bool('os' not in var_12.as_map['straight'])
    assert var_14 is True
    var_15 = var_12.lines_without_imports
    var_16 = bool(var_12.lines_without_imports == [])
    assert var_16 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = 'sections'
    var_8 = 'combine_as_imports'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = 'import os as os_system\n# comment\n'
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = var_12.as_map['straight']['os']
    var_14 = bool(var_12.as_map['straight']['os'] == ['os_system'])
    assert var_14 is True
    var_15 = var_12.lines_without_imports
    var_16 = bool(var_12.lines_without_imports == [])
    assert var_16 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = 'sections'
    var_8 = 'force_single_line'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = 'from os import path  # comment\n'
    var_12 = module_1.file_contents(var_11, var_10)
    var_13 = var_12.categorized_comments['nested']['os']['path']
    assert var_13 == ' comment'
    var_14 = var_12.lines_without_imports
    var_15 = bool(var_12.lines_without_imports == [])
    assert var_15 is True


def test_case_0():
    var_0 = 'FUTURE'
    var_1 = 'STDLIB'
    var_2 = 'THIRDPARTY'
    var_3 = 'FIRSTPARTY'
    var_4 = 'LOCALFOLDER'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = '# NOQA'
    var_7 = [var_6]
    var_8 = 'sections'
    var_9 = 'treat_comments_as_code'
    var_10 = {var_8: var_5, var_9: var_7}
    var_11 = module_0.Config(**var_10)



# Parsed testcases at query #4
#--------------------------





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
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------




import isort.parse as module_0


def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = ()
    var_3 = False
    var_4 = module_0.skip_line(var_0, var_0, var_1, var_2, var_3)
    var_5 = bool(not var_4[0])
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 1'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')"
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'def func(): pass'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "a = '''not a docstring'''"
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'b = """not a docstring"""'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from sys import path'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'class MyClass: pass'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'result = 42'
    var_5 = module_1.file_contents(var_4, var_3)


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = ''
    var_5 = module_1.file_contents(var_4, var_3)



# Parsed testcases at query #7
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


def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.categorized_comments['straight']['os']
    var_5 = bool(var_1.categorized_comments['straight']['os'] == [' comment'])
    assert var_5 is True


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


def test_case_0():
    var_0 = 'import numpy as np'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = var_1.as_map['straight']['numpy']
    var_5 = bool(var_1.as_map['straight']['numpy'] == ['np'])
    assert var_5 is True


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


def test_case_0():
    var_0 = '# STDLIB'
    var_1 = '# THIRDPARTY'
    var_2 = [var_0, var_1]
    var_3 = 'section_comments'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = '# STDLIB\nimport os\n# THIRDPARTY\nimport numpy'
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
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.place_imports['STDLIB']
    var_5 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_5 is True


def test_case_0():
    var_0 = 'from os import (path, sep,)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0


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
    var_2 = var_1.lines_without_imports
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_1.lines_without_imports[0]
    assert var_4 == 'import os  # isort:skip'

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pd\nimport pandas as pd2'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' in var_5.imports['THIRDPARTY']['straight'])
    assert var_7 is True
    var_8 = var_5.as_map['straight']['pandas']
    var_9 = bool(var_5.as_map['straight']['pandas'] == ['pd', 'pd2'])
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pandas'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' in var_5.imports['THIRDPARTY']['straight'])
    assert var_7 is True
    var_8 = 'pandas'
    var_9 = bool('pandas' not in var_5.as_map['straight'])
    assert var_9 is True


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


def test_case_0():
    var_0 = '# comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.change_count
    assert var_3 == 0


def test_case_0():
    var_0 = "import os\nprint('hello')"
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.lines_without_imports[-1]
    assert var_4 == "print('hello')"


def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True


def test_case_0():
    var_0 = 'from libc cimport math'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'libc'
    var_3 = bool('libc' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'math'
    var_5 = bool('math' in var_1.imports['THIRDPARTY']['from']['libc'])
    assert var_5 is True


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



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'


def test_case_0():
    var_0 = '\r\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\r\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\r\nimport sys\nimport json'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = 'isort:imports-future'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)


def test_case_0():
    var_0 = '# some other comment'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)


def test_case_0():
    var_0 = '  # isort:imports-future'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)


def test_case_0():
    var_0 = '#isort:imports-future'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)


def test_case_0():
    var_0 = '# isort: imports-future'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)



# Parsed testcases at query #11
#--------------------------




import isort.parse as module_0


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.imports['STDLIB']['straight']['sys']
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0


def test_case_0():
    var_0 = 'from os import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = var_1.imports['STDLIB']['from']['os']['path']
    assert var_6 is True


def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.categorized_comments['straight']['os']
    var_4 = bool(var_1.categorized_comments['straight']['os'] == ['comment'])
    assert var_4 is True


def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)'
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


def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.as_map['straight']['os']
    var_4 = bool(var_1.as_map['straight']['os'] == ['operating_system'])
    assert var_4 is True


def test_case_0():
    var_0 = 'from os import path as p'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True
    var_6 = var_1.as_map['from']['os.path']
    var_7 = bool(var_1.as_map['from']['os.path'] == ['p'])
    assert var_7 is True


def test_case_0():
    var_0 = 'import os\n# isort: skip\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = 'sys'
    var_4 = bool('sys' not in var_1.imports['STDLIB']['straight'])
    assert var_4 is True


def test_case_0():
    var_0 = '# isort: imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'STDLIB'
    var_3 = bool('STDLIB' in var_1.place_imports)
    assert var_3 is True
    var_4 = var_1.imports['STDLIB']['straight']['os']
    assert var_4 is True


def test_case_0():
    var_0 = 'from os import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.trailing_commas)
    assert var_3 is True

import isort.settings as module_0


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport sys'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'os'
    var_8 = bool('os' in var_6.imports['os']['straight'])
    assert var_8 is True
    var_9 = 'sys'
    var_10 = bool('sys' in var_6.imports['STDLIB']['straight'])
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


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


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as o\nimport sys as s'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_7 is True
    var_8 = 'sys'
    var_9 = bool('sys' in var_5.imports['STDLIB']['straight'])
    assert var_9 is True


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


def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1


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

import isort.parse as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1
    var_3 = var_1.imports
    var_4 = len(var_3)
    assert var_4 == 0


def test_case_0():
    var_0 = '# comment\n# another'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.import_index
    assert var_2 == -1


def test_case_0():
    var_0 = 'from os import \\\n    path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['os'])
    assert var_5 is True


def test_case_0():
    var_0 = 'from libc cimport math'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'libc'
    var_3 = bool('libc' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'math'
    var_5 = bool('math' in var_1.imports['THIRDPARTY']['from']['libc'])
    assert var_5 is True


def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.imports['STDLIB']['straight']['os']
    assert var_2 is True
    var_3 = var_1.imports['STDLIB']['straight']['sys']
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_241_evaluates_to_true. Retrieved 3/4 statements.


import isort.settings as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something as alias'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = 'as'
    var_5 = bool('as' in var_3['imports']['from']['module']['something'])
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['straight']['os']
    assert var_4 is True
    var_5 = var_3.imports['STDLIB']['straight']['sys']
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == 0
    var_7 = var_3.change_count
    assert var_7 == 0


def test_case_0():
    var_0 = 'from os import path'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['from']['os']['path']
    assert var_4 is True
    var_5 = var_3.import_index
    assert var_5 == 0
    var_6 = var_3.change_count
    assert var_6 == 0


def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['straight']['os']
    assert var_4 is True
    var_5 = var_3.import_index
    assert var_5 == 1
    var_6 = var_3.change_count
    assert var_6 == 0


def test_case_0():
    var_0 = 'from os import (\n    path,\n    sep\n)'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['from']['os']['path']
    assert var_4 is True
    var_5 = var_3.imports['STDLIB']['from']['os']['sep']
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == 0
    var_7 = var_3.change_count
    assert var_7 == 0


def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['straight']['os']
    assert var_4 is True
    var_5 = var_3.as_map['straight']['os']
    var_6 = bool(var_3.as_map['straight']['os'] == ['operating_system'])
    assert var_6 is True
    var_7 = var_3.import_index
    assert var_7 == 0
    var_8 = var_3.change_count
    assert var_8 == 0


def test_case_0():
    var_0 = 'from os import path as p'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['from']['os']['path']
    assert var_4 is True
    var_5 = var_3.as_map['from']['os.path']
    var_6 = bool(var_3.as_map['from']['os.path'] == ['p'])
    assert var_6 is True
    var_7 = var_3.import_index
    assert var_7 == 0
    var_8 = var_3.change_count
    assert var_8 == 0


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport sys'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.imports['os']['straight']['os']
    assert var_7 is True
    var_8 = var_6.imports['STDLIB']['straight']['sys']
    assert var_8 is True
    var_9 = var_6.import_index
    assert var_9 == 0
    var_10 = var_6.change_count
    assert var_10 == 0


def test_case_0():
    var_0 = '# standard library'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# standard library\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.imports['STDLIB']['straight']['os']
    assert var_7 is True
    var_8 = var_6.import_index
    assert var_8 == 1
    var_9 = var_6.change_count
    assert var_9 == 0


def test_case_0():
    var_0 = 'from os import path,'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['from']['os']['path']
    assert var_4 is True
    var_5 = 'os'
    var_6 = bool('os' in var_3.trailing_commas)
    assert var_6 is True
    var_7 = var_3.import_index
    assert var_7 == 0
    var_8 = var_3.change_count
    assert var_8 == 0


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
    var_9 = 'else-type place_module for os returned'
    var_10 = bool('else-type place_module for os returned' in var_5.verbose_output[0])
    assert var_10 is True
    var_11 = var_5.import_index
    assert var_11 == 0
    var_12 = var_5.change_count
    assert var_12 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.imports['STDLIB']['straight']['os']
    assert var_6 is True
    var_7 = var_5.import_index
    assert var_7 == 0
    var_8 = var_5.change_count
    assert var_8 == 0


def test_case_0():
    var_0 = '# isort:imports-STDLIB\nimport os'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['straight']['os']
    assert var_4 is True
    var_5 = var_3.import_placements['# isort:imports-STDLIB']
    assert var_5 == 'STDLIB'
    var_6 = var_3.import_index
    assert var_6 == 1
    var_7 = var_3.change_count
    assert var_7 == 0


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path as p\n# comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.imports['STDLIB']['from']['os']['path']
    assert var_6 is True
    var_7 = var_5.as_map['from']['os.path']
    var_8 = bool(var_5.as_map['from']['os.path'] == ['p'])
    assert var_8 is True
    var_9 = var_5.import_index
    assert var_9 == 0
    var_10 = var_5.change_count
    assert var_10 == 0


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.imports['STDLIB']['straight']['os']
    assert var_6 is True
    var_7 = 'os'
    var_8 = bool('os' not in var_5.as_map['straight'])
    assert var_8 is True
    var_9 = var_5.import_index
    assert var_9 == 0
    var_10 = var_5.change_count
    assert var_10 == 0


def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.imports['STDLIB']['straight']['os']
    assert var_7 is True
    var_8 = var_6.import_index
    assert var_8 == 1
    var_9 = var_6.change_count
    assert var_9 == 0


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True
    var_7 = var_3.import_index
    assert var_7 == -1
    var_8 = var_3.change_count
    assert var_8 == 0


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
    var_10 = 'else-type place_module for os returned'
    var_11 = bool('else-type place_module for os returned' in var_6.verbose_output[0])
    assert var_11 is True
    var_12 = var_6.import_index
    assert var_12 == 0
    var_13 = var_6.change_count
    assert var_13 == 0


def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.imports['STDLIB']['from']['os']['path']
    assert var_6 is True
    var_7 = var_5.import_index
    assert var_7 == 0
    var_8 = var_5.change_count
    assert var_8 == 0


def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['STDLIB']['straight']['os']
    assert var_4 is True
    var_5 = var_3.imports['STDLIB']['straight']['sys']
    assert var_5 is True
    var_6 = var_3.import_index
    assert var_6 == 0
    var_7 = var_3.change_count
    assert var_7 == 0


def test_case_0():
    var_0 = 'from libc cimport math'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = var_3.imports['THIRDPARTY']['from']['libc']['math']
    assert var_4 is True
    var_5 = var_3.import_index
    assert var_5 == 0
    var_6 = var_3.change_count
    assert var_6 == 0


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


def test_case_0():
    var_0 = 'import os\r\nimport sys'
    var_1 = None
    var_2 = 'line_ending'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1.file_contents(var_0, var_4)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'
    var_7 = var_5.import_index
    assert var_7 == 0
    var_8 = var_5.change_count
    assert var_8 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.parse as module_0


def test_case_0():
    var_0 = 'import numpy'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy'


def test_case_0():
    var_0 = 'from numpy import array'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy array'


def test_case_0():
    var_0 = 'import numpy, pandas, sklearn'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy pandas sklearn'


def test_case_0():
    var_0 = 'from numpy import (array, matrix)'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy array matrix'


def test_case_0():
    var_0 = 'from numpy import array, \\\n    matrix, \\\n    linalg'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy array matrix linalg'


def test_case_0():
    var_0 = 'from libc.math cimport sin'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'libc.math sin'


def test_case_0():
    var_0 = 'from numpy import { array, matrix }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy {|array matrix|}'


def test_case_0():
    var_0 = 'from numpy import { array , matrix }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy {|array matrix|}'


def test_case_0():
    var_0 = 'from numpy import (array, { matrix, linalg })'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy array {|matrix linalg|}'


def test_case_0():
    var_0 = 'from numpy import { array, \\\n    matrix }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'numpy {|array matrix|}'


def test_case_0():
    var_0 = 'from module import _import'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module _import'


def test_case_0():
    var_0 = 'from module import _cimport'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'module _cimport'


def test_case_0():
    var_0 = 'from libc.math cimport (sin, cos, \\\n    tan), from numpy import { array, matrix }'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'libc.math sin cos tan numpy {|array matrix|}'


def test_case_0():
    var_0 = ''
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''


def test_case_0():
    var_0 = 'from import cimport'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == ''



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_file_contents_basic_imports. Retrieved 8/13 statements.
# Partially parsed test_file_contents_from_import. Retrieved 11/21 statements.
# Partially parsed test_file_contents_mixed_imports. Retrieved 13/19 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 10/15 statements.
# Partially parsed test_file_contents_with_aliases. Retrieved 10/15 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 9/13 statements.
# Partially parsed test_file_contents_forced_separate. Retrieved 11/13 statements.
# Partially parsed test_file_contents_section_comments. Retrieved 13/15 statements.
# Partially parsed test_file_contents_float_to_top. Retrieved 7/8 statements.
# Partially parsed test_file_contents_trailing_comma. Retrieved 7/11 statements.
# Partially parsed test_file_contents_empty_file. Retrieved 2/7 statements.
# Partially parsed test_file_contents_only_comments. Retrieved 2/7 statements.
# Partially parsed test_file_contents_with_code_after_imports. Retrieved 6/7 statements.
# Partially parsed test_file_contents_import_with_backslash. Retrieved 9/13 statements.
# Partially parsed test_file_contents_import_with_semicolon. Retrieved 8/9 statements.
# Partially parsed test_file_contents_isort_directives. Retrieved 9/11 statements.



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


def test_case_0():
    var_0 = 'from os import path\nfrom sys import modules\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['from']
    var_4 = 'os'
    var_5 = 'path'
    var_6 = True
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = 'sys'
    var_11 = 'modules'
    var_12 = (var_11, var_6)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = var_1.imports['STDLIB']['from']
    var_16 = []
    var_17 = var_1.imports['THIRDPARTY']['from']
    var_18 = []
    var_19 = var_1.imports['FIRSTPARTY']['from']
    var_20 = []
    var_21 = var_1.imports['LOCALFOLDER']['from']
    var_22 = var_1.import_index
    assert var_22 == 0
    var_23 = var_1.change_count
    assert var_23 == 0


def test_case_0():
    var_0 = 'import os\nfrom sys import modules\nimport numpy\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = 'sys'
    var_9 = 'modules'
    var_10 = (var_9, var_3)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = var_1.imports['STDLIB']['from']
    var_14 = 'numpy'
    var_15 = (var_14, var_3)
    var_16 = [var_15]
    var_17 = [var_16]
    var_18 = var_1.imports['THIRDPARTY']['straight']
    var_19 = var_1.import_index
    assert var_19 == 0
    var_20 = var_1.change_count
    assert var_20 == 0


def test_case_0():
    var_0 = 'import os  # comment\nfrom sys import modules  # another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = 'sys'
    var_9 = 'modules'
    var_10 = (var_9, var_3)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = var_1.imports['STDLIB']['from']
    var_14 = var_1.categorized_comments['straight']['os']
    var_15 = bool(var_1.categorized_comments['straight']['os'] == ['# comment'])
    assert var_15 is True
    var_16 = var_1.categorized_comments['from']['sys']
    var_17 = bool(var_1.categorized_comments['from']['sys'] == ['# another comment'])
    assert var_17 is True
    var_18 = var_1.import_index
    assert var_18 == 0
    var_19 = var_1.change_count
    assert var_19 == 0


def test_case_0():
    var_0 = 'import os as operating_system\nfrom sys import modules as mods\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = 'sys'
    var_9 = 'modules'
    var_10 = (var_9, var_3)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = var_1.imports['STDLIB']['from']
    var_14 = var_1.as_map['straight']['os']
    var_15 = bool(var_1.as_map['straight']['os'] == ['operating_system'])
    assert var_15 is True
    var_16 = var_1.as_map['from']['sys.modules']
    var_17 = bool(var_1.as_map['from']['sys.modules'] == ['mods'])
    assert var_17 is True
    var_18 = var_1.import_index
    assert var_18 == 0
    var_19 = var_1.change_count
    assert var_19 == 0


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
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0

import isort.settings as module_0


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport sys\n'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'sys'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = var_6.imports['STDLIB']['straight']
    var_13 = (var_0, var_8)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = var_6.imports['os']['straight']
    var_17 = var_6.import_index
    assert var_17 == 0
    var_18 = var_6.change_count
    assert var_18 == 0


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
    var_9 = True
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = var_7.imports['STDLIB']['straight']
    var_14 = 'numpy'
    var_15 = (var_14, var_9)
    var_16 = [var_15]
    var_17 = [var_16]
    var_18 = var_7.imports['THIRDPARTY']['straight']
    var_19 = var_7.import_index
    assert var_19 == 0
    var_20 = var_7.change_count
    assert var_20 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = var_5.imports['STDLIB']['straight']
    var_11 = var_5.import_index
    assert var_11 == 0
    var_12 = var_5.change_count
    assert var_12 == 0

import isort.parse as module_0


def test_case_0():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'path'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']
    var_9 = 'os'
    var_10 = bool('os' in var_1.trailing_commas)
    assert var_10 is True
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0


def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = []
    var_5 = var_1.imports['STDLIB']['straight']
    var_6 = []
    var_7 = var_1.imports['THIRDPARTY']['straight']
    var_8 = []
    var_9 = var_1.imports['FIRSTPARTY']['straight']
    var_10 = []
    var_11 = var_1.imports['LOCALFOLDER']['straight']
    var_12 = var_1.import_index
    assert var_12 == -1
    var_13 = var_1.change_count
    assert var_13 == 0


def test_case_0():
    var_0 = '# comment\n# another comment\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = []
    var_5 = var_1.imports['STDLIB']['straight']
    var_6 = []
    var_7 = var_1.imports['THIRDPARTY']['straight']
    var_8 = []
    var_9 = var_1.imports['FIRSTPARTY']['straight']
    var_10 = []
    var_11 = var_1.imports['LOCALFOLDER']['straight']
    var_12 = var_1.import_index
    assert var_12 == -1
    var_13 = var_1.change_count
    assert var_13 == 0


def test_case_0():
    var_0 = "import os\nprint('hello')\n"
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.lines_without_imports
    var_9 = bool(var_1.lines_without_imports == ["print('hello')"])
    assert var_9 is True
    var_10 = var_1.import_index
    assert var_10 == 0
    var_11 = var_1.change_count
    assert var_11 == -1


def test_case_0():
    var_0 = 'from os import path, \\\nsep\n'
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
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0


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
    var_10 = var_1.import_index
    assert var_10 == 0
    var_11 = var_1.change_count
    assert var_11 == 0


def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os\n# isort:imports-thirdparty\nimport numpy\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = 'numpy'
    var_9 = (var_8, var_3)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = var_1.imports['THIRDPARTY']['straight']
    var_13 = var_1.place_imports['STDLIB']
    var_14 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_14 is True
    var_15 = var_1.place_imports['THIRDPARTY']
    var_16 = bool(var_1.place_imports['THIRDPARTY'] == [])
    assert var_16 is True
    var_17 = var_1.import_index
    assert var_17 == 0
    var_18 = var_1.change_count
    assert var_18 == 0

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\n'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.verbose_output
    var_7 = bool(var_5.verbose_output == ['else-type place_module for os returned STDLIB'])
    assert var_7 is True
    var_8 = var_5.import_index
    assert var_8 == 0
    var_9 = var_5.change_count
    assert var_9 == 0


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os\nfrom sys import modules as modules\n'
    var_5 = module_1.file_contents(var_4, var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 13/27 statements.



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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_392_evaluates_to_true. Retrieved 13/27 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n# comment\nimport sys'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.out_lines
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = '#'
    var_8 = '"""'
    var_9 = "'''"
    var_10 = 'isort:imports-'
    var_11 = 'isort: imports-'
    var_12 = var_1.treat_all_comments_as_code
    var_13 = var_1.treat_comments_as_code



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_import_string_contains_import_after_replacements. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'from module import'
    var_1 = 'import('
    var_2 = 'import ('
    var_3 = '\\'
    var_4 = ' '
    var_5 = '\n'
    var_6 = 'import '
    var_7 = bool('import ' in var_0)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_file_contents_basic_import. Retrieved 5/10 statements.
# Partially parsed test_file_contents_from_import. Retrieved 6/13 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 6/7 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 5/6 statements.
# Partially parsed test_file_contents_forced_separate. Retrieved 9/11 statements.
# Partially parsed test_file_contents_as_import. Retrieved 5/6 statements.
# Partially parsed test_file_contents_from_import_with_as. Retrieved 6/9 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 7/10 statements.
# Partially parsed test_file_contents_with_section_comments. Retrieved 8/9 statements.
# Partially parsed test_file_contents_isort_skip. Retrieved 5/6 statements.
# Partially parsed test_file_contents_empty_file. Retrieved 2/7 statements.
# Partially parsed test_file_contents_with_backslash_continuation. Retrieved 6/9 statements.
# Partially parsed test_file_contents_cimport. Retrieved 6/9 statements.
# Partially parsed test_file_contents_semicolon_separated. Retrieved 6/7 statements.


import isort.parse as module_0


def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = 'os'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['straight']
    var_9 = []
    var_10 = var_1.imports['THIRDPARTY']['straight']
    var_11 = []
    var_12 = var_1.imports['FIRSTPARTY']['straight']
    var_13 = []
    var_14 = var_1.imports['LOCALFOLDER']['straight']
    var_15 = var_1.import_index
    assert var_15 == 0
    var_16 = var_1.change_count
    assert var_16 == 0


def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['from']
    var_4 = 'sys'
    var_5 = 'path'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = [var_7]
    var_9 = var_1.imports['STDLIB']['from']
    var_10 = []
    var_11 = var_1.imports['THIRDPARTY']['from']
    var_12 = []
    var_13 = var_1.imports['FIRSTPARTY']['from']
    var_14 = []
    var_15 = var_1.imports['LOCALFOLDER']['from']
    var_16 = var_1.import_index
    assert var_16 == 0


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = True
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.import_index
    assert var_8 == 0


def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = var_1.imports['STDLIB']['straight']
    var_7 = var_1.categorized_comments['straight']['os']
    var_8 = bool(var_1.categorized_comments['straight']['os'] == [' comment'])
    assert var_8 is True

import isort.settings as module_0


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport sys'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = True
    var_8 = {var_0: var_7}
    var_9 = [var_8]
    var_10 = var_6.imports['os']['straight']
    var_11 = 'sys'
    var_12 = {var_11: var_7}
    var_13 = [var_12]
    var_14 = var_6.imports['STDLIB']['straight']

import isort.parse as module_0


def test_case_0():
    var_0 = 'import os as operating_system'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = var_1.imports['STDLIB']['straight']
    var_7 = var_1.as_map['straight']['os']
    var_8 = bool(var_1.as_map['straight']['os'] == ['operating_system'])
    assert var_8 is True


def test_case_0():
    var_0 = 'from sys import path as sys_path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['from']
    var_8 = var_1.as_map['from']['sys.path']
    var_9 = bool(var_1.as_map['from']['sys.path'] == ['sys_path'])
    assert var_9 is True


def test_case_0():
    var_0 = 'from sys import (\n    path,\n    argv\n)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = 'argv'
    var_5 = True
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']

import isort.settings as module_0


def test_case_0():
    var_0 = '# stdlib'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# stdlib\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'os'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = var_6.imports['STDLIB']['straight']

import isort.parse as module_0


def test_case_0():
    var_0 = 'from sys import path,'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.trailing_commas
    var_3 = bool(var_1.trailing_commas == {'sys'})
    assert var_3 is True

import isort.settings as module_0


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
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = var_1.imports['STDLIB']['straight']


def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.place_imports['STDLIB']
    var_3 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_3 is True
    var_4 = var_1.import_placements['# isort:imports-stdlib']
    assert var_4 == 'STDLIB'

import isort.settings as module_0


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


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.as_map['straight']['os']
    var_7 = bool(var_5.as_map['straight']['os'] == [])
    assert var_7 is True


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from sys import path as sys_path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['from']['sys.__combined_as__']
    var_7 = bool(var_5.categorized_comments['from']['sys.__combined_as__'] == [' comment'])
    assert var_7 is True


def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = var_6.import_index
    assert var_7 == 1

import isort.parse as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = []
    var_5 = var_1.imports['STDLIB']['straight']
    var_6 = []
    var_7 = var_1.imports['THIRDPARTY']['straight']
    var_8 = []
    var_9 = var_1.imports['FIRSTPARTY']['straight']
    var_10 = []
    var_11 = var_1.imports['LOCALFOLDER']['straight']
    var_12 = var_1.import_index
    assert var_12 == -1

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'verbose'
    var_2 = 'only_modified'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'else-type place_module for os returned STDLIB'
    var_8 = bool('else-type place_module for os returned STDLIB' in var_6.verbose_output)
    assert var_8 is True


def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from sys import path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.categorized_comments['nested']['sys']['path']
    assert var_6 == ' comment'


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

import isort.parse as module_0


def test_case_0():
    var_0 = 'import os\r\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = var_1.line_separator
    assert var_2 == '\r\n'


def test_case_0():
    var_0 = 'from sys import \\\n    path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['from']


def test_case_0():
    var_0 = 'from sys cimport path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['from']


def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = 'sys'
    var_4 = True
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('Hello, world!')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "    print('Hello, world!')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "\n\nprint('Hello, world!')"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "\n\nprint('Hello, world!')\n\n"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '# This is a comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '"""This is a docstring"""'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "'''This is a docstring'''"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'in_quote = "some string"'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == -1


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'x = 5'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'def foo(): pass'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'class Bar: pass'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.import_index
    assert var_6 == 0



# Parsed testcases at query #8
#--------------------------




import isort.parse as module_0


def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'


def test_case_0():
    var_0 = 'cimport numpy'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'


def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'from'

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # noqa'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None


def test_case_0():
    var_0 = True
    var_1 = 'honor_noqa'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os  # NOQA'
    var_5 = module_1.import_type(var_4, var_3)
    assert var_5 is None


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


def test_case_0():
    var_0 = 'import os  # isort: skip'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = 'import os  # isort: split'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = "print('Hello')"
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = 'from_import something'
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = ''
    var_1 = module_0.import_type(var_0)
    assert var_1 is None


def test_case_0():
    var_0 = '   '
    var_1 = module_0.import_type(var_0)
    assert var_1 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_file_contents_empty_file. Retrieved 2/3 statements.
# Partially parsed test_file_contents_only_comments. Retrieved 2/3 statements.



def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 0
    var_5 = var_1.change_count
    assert var_5 == 0


def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = bool('sys' in var_1.imports['STDLIB']['from'])
    assert var_3 is True
    var_4 = 'path'
    var_5 = bool('path' in var_1.imports['STDLIB']['from']['sys'])
    assert var_5 is True
    var_6 = var_1.import_index
    assert var_6 == 0


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


def test_case_0():
    var_0 = '# comment\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = var_1.import_index
    assert var_4 == 1


def test_case_0():
    var_0 = 'import pandas as pd'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'pandas'
    var_3 = bool('pandas' in var_1.imports['THIRDPARTY']['straight'])
    assert var_3 is True
    var_4 = 'pd'
    var_5 = bool('pd' in var_1.as_map['straight']['pandas'])
    assert var_5 is True


def test_case_0():
    var_0 = 'from numpy import array as arr'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'numpy'
    var_3 = bool('numpy' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'arr'
    var_5 = bool('arr' in var_1.as_map['from']['numpy.array'])
    assert var_5 is True


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


def test_case_0():
    var_0 = 'from os.path import join, split,'
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
    var_8 = 'os.path'
    var_9 = bool('os.path' in var_1.trailing_commas)
    assert var_9 is True


def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'STDLIB'
    var_5 = bool('STDLIB' in var_1.place_imports)
    assert var_5 is True

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_7 is True
    var_8 = var_5.import_index
    assert var_8 == 0


def test_case_0():
    var_0 = True
    var_1 = 'force_single_line'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from os import path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['from'])
    assert var_7 is True
    var_8 = 'path'
    var_9 = bool('path' in var_5.imports['STDLIB']['from']['os'])
    assert var_9 is True
    var_10 = 'comment'
    var_11 = bool('comment' in var_5.categorized_comments['nested']['os']['path'])
    assert var_11 is True


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = bool('os' in var_5.imports['STDLIB']['straight'])
    assert var_7 is True
    var_8 = 'os'
    var_9 = bool('os' not in var_5.as_map['straight'])
    assert var_9 is True


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import pandas as pd  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'pandas'
    var_7 = bool('pandas' in var_5.imports['THIRDPARTY']['straight'])
    assert var_7 is True
    var_8 = 'pd'
    var_9 = bool('pd' in var_5.as_map['straight']['pandas'])
    assert var_9 is True
    var_10 = 'comment'
    var_11 = bool('comment' in var_5.categorized_comments['straight']['pandas'])
    assert var_11 is True


def test_case_0():
    var_0 = '# noqa'
    var_1 = [var_0]
    var_2 = 'treat_comments_as_code'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# noqa\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'os'
    var_8 = bool('os' in var_6.imports['STDLIB']['straight'])
    assert var_8 is True
    var_9 = var_6.import_index
    assert var_9 == 1

import isort.parse as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.import_index
    assert var_4 == -1


def test_case_0():
    var_0 = '# comment\n# another'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports
    var_4 = var_1.import_index
    assert var_4 == -1


def test_case_0():
    var_0 = 'import os; import sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = 'sys'
    var_5 = bool('sys' in var_1.imports['STDLIB']['straight'])
    assert var_5 is True


def test_case_0():
    var_0 = 'from cython cimport boundscheck'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'cython'
    var_3 = bool('cython' in var_1.imports['THIRDPARTY']['from'])
    assert var_3 is True
    var_4 = 'boundscheck'
    var_5 = bool('boundscheck' in var_1.imports['THIRDPARTY']['from']['cython'])
    assert var_5 is True

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'verbose'
    var_3 = 'only_modified'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'import os'
    var_7 = module_1.file_contents(var_6, var_5)
    var_8 = 'else-type place_module for os returned STDLIB'
    var_9 = bool('else-type place_module for os returned STDLIB' in var_7.verbose_output)
    assert var_9 is True


def test_case_0():
    var_0 = 'FIRSTPARTY'
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
    var_0 = '# above\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = bool('os' in var_1.imports['STDLIB']['straight'])
    assert var_3 is True
    var_4 = '# above'
    var_5 = bool('# above' in var_1.categorized_comments['above']['straight']['os'])
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_file_contents_predicate_false. Retrieved 3/9 statements.


import isort.settings as module_0


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = {}
    var_1 = 'some_module'
    var_2 = 'func1'
    var_3 = 'func2'
    var_4 = [var_2, var_3]
    var_5 = [var_2]
    var_6 = var_1 not in var_0



# Parsed testcases at query #12
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'from module import something  # comment'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.imports['from']['module']['something']
    var_5 = bool(var_3.imports['from']['module']['something'] is not None)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_340_evaluates_to_true. Retrieved 15/23 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'import os\n# comment\nfrom sys import path'
    var_3 = module_1.file_contents(var_2, var_1)
    var_4 = var_3.out_lines
    var_5 = len(var_4)
    var_6 = var_3.import_index
    var_7 = 1
    var_8 = max(var_6, var_7)
    var_9 = var_8 - var_7
    var_10 = bool(var_5 > var_9)
    assert var_10 is True
    var_11 = -1
    var_12 = var_4[var_11]
    var_13 = ''
    var_14 = '#'
    var_15 = '"""'
    var_16 = "'''"
    var_17 = 'isort:imports-'
    var_18 = 'isort: imports-'
    var_19 = bool(not var_1.treat_all_comments_as_code)
    assert var_19 is True



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_file_contents_basic_import. Retrieved 6/11 statements.
# Partially parsed test_file_contents_from_import. Retrieved 7/15 statements.
# Partially parsed test_file_contents_multiple_imports. Retrieved 8/9 statements.
# Partially parsed test_file_contents_with_comments. Retrieved 6/7 statements.
# Partially parsed test_file_contents_with_aliases. Retrieved 6/7 statements.
# Partially parsed test_file_contents_from_import_with_aliases. Retrieved 7/11 statements.
# Partially parsed test_file_contents_multiline_import. Retrieved 9/13 statements.
# Partially parsed test_file_contents_forced_separate. Retrieved 11/13 statements.
# Partially parsed test_file_contents_section_comments. Retrieved 9/10 statements.
# Partially parsed test_file_contents_trailing_comma. Retrieved 7/11 statements.
# Partially parsed test_file_contents_float_to_top. Retrieved 7/8 statements.
# Partially parsed test_file_contents_isort_skip. Retrieved 6/7 statements.
# Partially parsed test_file_contents_isort_imports_section. Retrieved 6/7 statements.
# Partially parsed test_file_contents_combined_as_imports. Retrieved 8/12 statements.
# Partially parsed test_file_contents_remove_redundant_aliases. Retrieved 7/8 statements.
# Partially parsed test_file_contents_treat_all_comments_as_code. Retrieved 7/8 statements.
# Partially parsed test_file_contents_empty_file. Retrieved 2/7 statements.
# Partially parsed test_file_contents_newline_at_end. Retrieved 6/7 statements.
# Partially parsed test_file_contents_with_backslash_continuation. Retrieved 9/13 statements.


import isort.parse as module_0


def test_case_0():
    var_0 = 'import os'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = 'os'
    var_5 = True
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_7]
    var_9 = var_1.imports['STDLIB']['straight']
    var_10 = []
    var_11 = var_1.imports['THIRDPARTY']['straight']
    var_12 = []
    var_13 = var_1.imports['FIRSTPARTY']['straight']
    var_14 = []
    var_15 = var_1.imports['LOCALFOLDER']['straight']
    var_16 = var_1.import_index
    assert var_16 == 0
    var_17 = var_1.change_count
    assert var_17 == 0


def test_case_0():
    var_0 = 'from sys import path'
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['from']
    var_4 = 'sys'
    var_5 = 'path'
    var_6 = True
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = var_1.imports['STDLIB']['from']
    var_11 = []
    var_12 = var_1.imports['THIRDPARTY']['from']
    var_13 = []
    var_14 = var_1.imports['FIRSTPARTY']['from']
    var_15 = []
    var_16 = var_1.imports['LOCALFOLDER']['from']
    var_17 = var_1.import_index
    assert var_17 == 0
    var_18 = var_1.change_count
    assert var_18 == 0


def test_case_0():
    var_0 = 'import os\nimport sys'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = 'sys'
    var_6 = (var_5, var_3)
    var_7 = [var_4, var_6]
    var_8 = [var_7]
    var_9 = var_1.imports['STDLIB']['straight']
    var_10 = var_1.import_index
    assert var_10 == 0
    var_11 = var_1.change_count
    assert var_11 == 0


def test_case_0():
    var_0 = 'import os  # comment'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.categorized_comments['straight']['os']
    var_9 = bool(var_1.categorized_comments['straight']['os'] == ['comment'])
    assert var_9 is True
    var_10 = var_1.import_index
    assert var_10 == 0
    var_11 = var_1.change_count
    assert var_11 == 0


def test_case_0():
    var_0 = 'import os as operating_system'
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
    var_10 = var_1.import_index
    assert var_10 == 0
    var_11 = var_1.change_count
    assert var_11 == 0


def test_case_0():
    var_0 = 'from sys import path as sys_path'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'sys'
    var_3 = 'path'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']
    var_9 = var_1.as_map['from']['sys.path']
    var_10 = bool(var_1.as_map['from']['sys.path'] == ['sys_path'])
    assert var_10 is True
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0


def test_case_0():
    var_0 = 'from os.path import (join, split)'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = 'split'
    var_7 = (var_6, var_4)
    var_8 = [var_5, var_7]
    var_9 = [var_8]
    var_10 = var_1.imports['STDLIB']['from']
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0

import isort.settings as module_0


def test_case_0():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os\nimport sys'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = True
    var_8 = (var_0, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = var_6.imports['os']['straight']
    var_12 = 'sys'
    var_13 = (var_12, var_7)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = var_6.imports['STDLIB']['straight']
    var_17 = var_6.import_index
    assert var_17 == 0
    var_18 = var_6.change_count
    assert var_18 == 0


def test_case_0():
    var_0 = '# stdlib'
    var_1 = [var_0]
    var_2 = 'section_comments'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '# stdlib\nimport os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'os'
    var_8 = True
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = var_6.imports['STDLIB']['straight']
    var_13 = var_6.import_index
    assert var_13 == 1
    var_14 = var_6.change_count
    assert var_14 == 0

import isort.parse as module_0


def test_case_0():
    var_0 = 'from os.path import join,'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = [var_6]
    var_8 = var_1.imports['STDLIB']['from']
    var_9 = 'os.path'
    var_10 = bool('os.path' in var_1.trailing_commas)
    assert var_10 is True
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'float_to_top'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = "print('hello')\nimport os"
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = var_5.imports['STDLIB']['straight']
    var_11 = var_5.import_index
    assert var_11 == 0
    var_12 = var_5.change_count
    assert var_12 == 0

import isort.parse as module_0


def test_case_0():
    var_0 = 'import os  # isort:skip'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.import_index
    assert var_8 == 0
    var_9 = var_1.change_count
    assert var_9 == 0


def test_case_0():
    var_0 = '# isort:imports-stdlib\nimport os'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.place_imports['STDLIB']
    var_9 = bool(var_1.place_imports['STDLIB'] == [])
    assert var_9 is True
    var_10 = var_1.import_placements['# isort:imports-stdlib']
    assert var_10 == 'STDLIB'
    var_11 = var_1.import_index
    assert var_11 == 0
    var_12 = var_1.change_count
    assert var_12 == 0

import isort.settings as module_0


def test_case_0():
    var_0 = True
    var_1 = 'combine_as_imports'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'from sys import path as sys_path  # comment'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'sys'
    var_7 = 'path'
    var_8 = (var_7, var_0)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = var_5.imports['STDLIB']['from']
    var_12 = var_5.as_map['from']['sys.path']
    var_13 = bool(var_5.as_map['from']['sys.path'] == ['sys_path'])
    assert var_13 is True
    var_14 = var_5.categorized_comments['from']['sys.__combined_as__']
    var_15 = bool(var_5.categorized_comments['from']['sys.__combined_as__'] == ['comment'])
    assert var_15 is True
    var_16 = var_5.import_index
    assert var_16 == 0
    var_17 = var_5.change_count
    assert var_17 == 0


def test_case_0():
    var_0 = True
    var_1 = 'remove_redundant_aliases'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os as os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = var_5.imports['STDLIB']['straight']
    var_11 = var_5.as_map['straight']['os']
    var_12 = bool(var_5.as_map['straight']['os'] == [])
    assert var_12 is True
    var_13 = var_5.import_index
    assert var_13 == 0
    var_14 = var_5.change_count
    assert var_14 == 0


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
    var_8 = var_5.import_index
    assert var_8 == 0
    var_9 = var_5.change_count
    assert var_9 == 0


def test_case_0():
    var_0 = True
    var_1 = 'only_modified'
    var_2 = 'verbose'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_0.Config(**var_3)
    var_5 = 'import os'
    var_6 = module_1.file_contents(var_5, var_4)
    var_7 = 'else-type place_module for os returned STDLIB'
    var_8 = bool('else-type place_module for os returned STDLIB' in var_6.verbose_output)
    assert var_8 is True
    var_9 = var_6.import_index
    assert var_9 == 0
    var_10 = var_6.change_count
    assert var_10 == 0


def test_case_0():
    var_0 = True
    var_1 = 'treat_all_comments_as_code'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = '# comment\nimport os'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = 'os'
    var_7 = (var_6, var_0)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = var_5.imports['STDLIB']['straight']
    var_11 = var_5.lines_without_imports
    var_12 = bool(var_5.lines_without_imports == ['# comment'])
    assert var_12 is True
    var_13 = var_5.import_index
    assert var_13 == 1
    var_14 = var_5.change_count
    assert var_14 == 0

import isort.parse as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    var_2 = []
    var_3 = var_1.imports['FUTURE']['straight']
    var_4 = []
    var_5 = var_1.imports['STDLIB']['straight']
    var_6 = []
    var_7 = var_1.imports['THIRDPARTY']['straight']
    var_8 = []
    var_9 = var_1.imports['FIRSTPARTY']['straight']
    var_10 = []
    var_11 = var_1.imports['LOCALFOLDER']['straight']
    var_12 = var_1.import_index
    assert var_12 == -1
    var_13 = var_1.change_count
    assert var_13 == 0


def test_case_0():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os'
    var_3 = True
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = var_1.imports['STDLIB']['straight']
    var_8 = var_1.in_lines[-1]
    assert var_8 == ''
    var_9 = var_1.import_index
    assert var_9 == 0
    var_10 = var_1.change_count
    assert var_10 == 0


def test_case_0():
    var_0 = 'from os.path import join, \\\n    split'
    var_1 = module_0.file_contents(var_0)
    var_2 = 'os.path'
    var_3 = 'join'
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = 'split'
    var_7 = (var_6, var_4)
    var_8 = [var_5, var_7]
    var_9 = [var_8]
    var_10 = var_1.imports['STDLIB']['from']



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\n'


def test_case_0():
    var_0 = '\r\n'
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\rimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\r\nimport sys'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'


def test_case_0():
    var_0 = None
    var_1 = 'line_ending'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'import os\r\nimport sys\nimport json'
    var_5 = module_1.file_contents(var_4, var_3)
    var_6 = var_5.line_separator
    assert var_6 == '\r\n'



# Parsed testcases at query #17
#--------------------------





def test_case_0():
    var_0 = 'from module import (something, another_thing)'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = module_1.file_contents(var_0, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



