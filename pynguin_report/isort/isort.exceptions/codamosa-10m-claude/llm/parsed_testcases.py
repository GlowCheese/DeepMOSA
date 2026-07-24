####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipped exception initialization and attributes'
    var_1 = 'Test skip message'
    var_2 = '/path/to/file.py'
    var_3 = module_0.FileSkipped(var_1, var_2)
    var_4 = str(var_3)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipped with empty message'
    var_1 = ''
    var_2 = '/path/to/file.py'
    var_3 = module_0.FileSkipped(var_1, var_2)
    var_4 = str(var_3)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipped with special characters in paths and messages'
    var_1 = 'File skipped: \'special\' & "chars"'
    var_2 = '/path/to/file-with-dash_and_underscore.py'
    var_3 = module_0.FileSkipped(var_1, var_2)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that FileSkipped properly inherits from ISortError'
    var_1 = 'test'
    var_2 = 'file.py'
    var_3 = module_0.FileSkipped(var_1, var_2)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that FileSkipped can be pickled via __reduce__'
    var_1 = 'Skip message'
    var_2 = 'test.py'
    var_3 = module_0.FileSkipped(var_1, var_2)



# Parsed testcases at query #2
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test LiteralParsingFailure exception initialization and attributes'
    var_1 = '[1, 2, 3'
    var_2 = 'invalid syntax'
    var_3 = ValueError(var_2)
    var_4 = module_0.LiteralParsingFailure(var_1, var_3)
    var_5 = str(var_4)
    var_6 = str(var_4)
    var_7 = str(var_4)

def test_case_0():
    var_0 = 'Test LiteralParsingFailure with exception type instead of instance'
    var_1 = "{'key': value}"

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test LiteralParsingFailure can be pickled via __reduce__'
    var_1 = '(1, 2, 3'
    var_2 = 'unclosed parenthesis'
    var_3 = ValueError(var_2)
    var_4 = module_0.LiteralParsingFailure(var_1, var_3)



# Parsed testcases at query #3
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidSettingsPath exception initialization and attributes'
    var_1 = '/nonexistent/path'
    var_2 = module_0.InvalidSettingsPath(var_1)
    var_3 = str(var_2)
    var_4 = str(var_2)
    var_5 = str(var_2)
    var_6 = './config.ini'
    var_7 = module_0.InvalidSettingsPath(var_6)
    var_8 = str(var_7)



# Parsed testcases at query #4
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = '/invalid/path/to/settings'
    var_1 = module_0.InvalidSettingsPath(var_0)
    var_2 = str(var_1)
    var_3 = str(var_1)
    var_4 = str(var_1)



# Parsed testcases at query #5
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test SortingFunctionDoesNotExist exception initialization and attributes'
    var_1 = 'custom_sort'
    var_2 = 'natural'
    var_3 = 'length'
    var_4 = 'reverse'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.SortingFunctionDoesNotExist(var_1, var_5)
    var_7 = str(var_6)
    var_8 = str(var_6)
    var_9 = str(var_6)
    var_10 = str(var_6)
    var_11 = module_0.SortingFunctionDoesNotExist(var_1, var_5)



# Parsed testcases at query #6
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipComment exception initialization and attributes'
    var_1 = '/path/to/file.py'
    var_2 = module_0.FileSkipComment(var_1)
    var_3 = str(var_2)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipComment exception with additional kwargs'
    var_1 = '/path/to/another_file.py'
    var_2 = 'should_be_ignored'
    var_3 = module_0.FileSkipComment(var_1)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipComment exception can be pickled and unpickled'
    var_1 = '/path/to/file.py'
    var_2 = module_0.FileSkipComment(var_1)
    var_3 = type(var_2)
    var_4 = var_2.__dict__



# Parsed testcases at query #7
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test MissingSection exception initialization and attributes'
    var_1 = 'numpy'
    var_2 = 'CUSTOM'
    var_3 = module_0.MissingSection(var_1, var_2)
    var_4 = str(var_3)
    var_5 = str(var_3)
    var_6 = str(var_3)
    var_7 = str(var_3)
    var_8 = 'pandas'
    var_9 = 'MYLIB'
    var_10 = module_0.MissingSection(var_8, var_9)
    var_11 = str(var_10)
    var_12 = str(var_10)
    var_13 = 'requests'
    var_14 = 'EXTERNAL'
    var_15 = module_0.MissingSection(var_13, var_14)



# Parsed testcases at query #8
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipComment exception initialization and attributes'
    var_1 = '/path/to/file.py'
    var_2 = module_0.FileSkipComment(var_1)
    var_3 = str(var_2)
    var_4 = 'test.py'
    var_5 = module_0.FileSkipComment(var_4)
    var_6 = str(var_5)
    var_7 = 'value'
    var_8 = module_0.FileSkipComment(var_1)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidSettingsPath exception initialization and attributes'
    var_1 = '/non/existent/path'
    var_2 = module_0.InvalidSettingsPath(var_1)
    var_3 = str(var_2)
    var_4 = str(var_2)
    var_5 = str(var_2)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test InvalidSettingsPath exception pickling support via __reduce__'
    var_1 = '/some/path'
    var_2 = module_0.InvalidSettingsPath(var_1)
    var_3 = 0

def test_case_0():
    var_0 = 'Test InvalidSettingsPath with various path formats'
    var_1 = '/absolute/path'
    var_2 = 'relative/path'
    var_3 = '.'
    var_4 = 'C:\\Windows\\Path'
    var_5 = ''
    var_6 = [var_1, var_2, var_3, var_4, var_5]



# Parsed testcases at query #11
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnsupportedSettings exception initialization and behavior'
    var_1 = 'unknown_option'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'test_value'
    var_5 = 'config.ini'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.UnsupportedSettings(var_7)
    var_9 = str(var_8)
    var_10 = str(var_8)
    var_11 = str(var_8)
    var_12 = str(var_8)
    var_13 = str(var_8)
    var_14 = 'bad_option1'
    var_15 = 'bad_option2'
    var_16 = 'bad_option3'
    var_17 = 'value1'
    var_18 = 'cli'
    var_19 = {var_2: var_17, var_3: var_18}
    var_20 = 'value2'
    var_21 = 'environment'
    var_22 = {var_2: var_20, var_3: var_21}
    var_23 = 'value3'
    var_24 = 'runtime'
    var_25 = {var_2: var_23, var_3: var_24}
    var_26 = {var_14: var_19, var_15: var_22, var_16: var_25}
    var_27 = module_0.UnsupportedSettings(var_26)
    var_28 = str(var_27)
    var_29 = str(var_27)
    var_30 = str(var_27)
    var_31 = str(var_27)
    var_32 = str(var_27)
    var_33 = str(var_27)
    var_34 = str(var_27)
    var_35 = str(var_27)
    var_36 = str(var_27)
    var_37 = 'test_name'
    var_38 = 'test_val'
    var_39 = 'test_source'
    var_40 = {}
    var_41 = module_0.UnsupportedSettings(var_40)
    var_42 = str(var_41)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test that ISortError.__reduce__ returns the correct partial function and empty tuple'
    var_1 = 'Test error message'

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that __reduce__ works correctly with ISortError subclasses'
    var_1 = '/path/to/file.py'
    var_2 = module_0.ExistingSyntaxErrors(var_1)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that __reduce__ preserves all exception attributes'
    var_1 = 'option1'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'val1'
    var_5 = 'config'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.UnsupportedSettings(var_7)



# Parsed testcases at query #13
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'x = 1\ny = 2\nz = 3'
    var_1 = module_0.AssignmentsFormatMismatch(var_0)
    var_2 = str(var_1)
    var_3 = str(var_1)
    var_4 = str(var_1)
    var_5 = str(var_1)
    var_6 = str(var_1)
    var_7 = 'a = [1, 2, 3]'
    var_8 = module_0.AssignmentsFormatMismatch(var_7)
    var_9 = str(var_8)
    var_10 = module_0.AssignmentsFormatMismatch(var_0)



# Parsed testcases at query #14
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test FileSkipComment exception initialization and attributes'
    var_1 = '/path/to/file.py'
    var_2 = module_0.FileSkipComment(var_1)
    var_3 = str(var_2)
    var_4 = 'test.py'
    var_5 = module_0.FileSkipComment(var_4)
    var_6 = str(var_5)
    var_7 = 'value'
    var_8 = module_0.FileSkipComment(var_1)
    var_9 = str(var_8)



# Parsed testcases at query #15
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = module_0.UnsupportedEncoding(var_0)
    var_2 = str(var_1)
    assert var_2 == 'Unknown or unsupported encoding in test.py'
    var_3 = 'path/to/file.py'
    var_4 = str(var_1)
    var_5 = '/absolute/path/to/module.py'
    var_6 = module_0.UnsupportedEncoding(var_5)
    var_7 = str(var_6)
    var_8 = module_0.UnsupportedEncoding(var_0)
    var_9 = type(var_8)
    var_10 = var_8.__dict__



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test LiteralSortTypeMismatch exception initialization and attributes'

def test_case_0():
    var_0 = 'Test LiteralSortTypeMismatch with various type combinations'

def test_case_0():
    var_0 = 'Test that LiteralSortTypeMismatch error message is properly formatted'

def test_case_0():
    var_0 = 'Test that LiteralSortTypeMismatch properly inherits from ISortError'

def test_case_0():
    var_0 = 'Test that LiteralSortTypeMismatch can be pickled via __reduce__'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test LiteralParsingFailure exception initialization and attributes'
    var_1 = '[1, 2, 3'
    var_2 = 'invalid literal'
    var_3 = ValueError(var_2)
    var_4 = module_0.LiteralParsingFailure(var_1, var_3)
    var_5 = str(var_4)
    var_6 = str(var_4)
    var_7 = str(var_4)

def test_case_0():
    var_0 = 'Test LiteralParsingFailure with exception type instead of instance'
    var_1 = "{'key': value}"

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that LiteralParsingFailure can be pickled and unpickled'
    var_1 = '[1, 2, 3'
    var_2 = 'test error'
    var_3 = ValueError(var_2)
    var_4 = module_0.LiteralParsingFailure(var_1, var_3)
    var_5 = 0



# Parsed testcases at query #3
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'my_custom_formatter'
    var_1 = module_0.FormattingPluginDoesNotExist(var_0)
    var_2 = str(var_1)
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #4
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'invalid_literal'
    var_1 = "invalid literal for int() with base 10: 'invalid'"
    var_2 = ValueError(var_1)
    var_3 = module_0.LiteralParsingFailure(var_0, var_2)
    var_4 = str(var_3)
    var_5 = str(var_3)

def test_case_0():
    var_0 = '[1, 2, 3]'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test error'

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'test_code'
    var_1 = 'test'
    var_2 = ValueError(var_1)
    var_3 = module_0.LiteralParsingFailure(var_0, var_2)
    var_4 = 0



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test that ISortError.__reduce__ returns correct partial and empty tuple'
    var_1 = 'test message'

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that __reduce__ works correctly with ISortError subclasses'
    var_1 = '/nonexistent/path'
    var_2 = module_0.InvalidSettingsPath(var_1)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that __reduce__ preserves all instance attributes'
    var_1 = 'test_file.py'
    var_2 = module_0.ExistingSyntaxErrors(var_1)



# Parsed testcases at query #6
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test ProfileDoesNotExist exception initialization and attributes'
    var_1 = 'nonexistent_profile'
    var_2 = module_0.ProfileDoesNotExist(var_1)
    var_3 = str(var_2)
    var_4 = str(var_2)
    var_5 = 'custom_profile'
    var_6 = module_0.ProfileDoesNotExist(var_5)
    var_7 = str(var_6)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test that ISortError.__reduce__ returns proper pickle information'
    var_1 = 'test message'

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test that ISortError.__reduce__ preserves exception attributes'
    var_1 = '/nonexistent/path'
    var_2 = module_0.InvalidSettingsPath(var_1)

import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test __reduce__ with subclass having multiple attributes'
    var_1 = 'custom_sort'
    var_2 = 'sort_a'
    var_3 = 'sort_b'
    var_4 = [var_2, var_3]
    var_5 = module_0.SortingFunctionDoesNotExist(var_1, var_4)



# Parsed testcases at query #8
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnsupportedSettings exception initialization and error message formatting'
    var_1 = 'unknown_option'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'some_value'
    var_5 = 'config.ini'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.UnsupportedSettings(var_7)
    var_9 = str(var_8)
    var_10 = str(var_8)
    var_11 = str(var_8)
    var_12 = 'option1'
    var_13 = 'option2'
    var_14 = 'option3'
    var_15 = 'value1'
    var_16 = 'cli'
    var_17 = {var_2: var_15, var_3: var_16}
    var_18 = 'value2'
    var_19 = 'setup.cfg'
    var_20 = {var_2: var_18, var_3: var_19}
    var_21 = 123
    var_22 = 'pyproject.toml'
    var_23 = {var_2: var_21, var_3: var_22}
    var_24 = {var_12: var_17, var_13: var_20, var_14: var_23}
    var_25 = module_0.UnsupportedSettings(var_24)
    var_26 = str(var_25)
    var_27 = {}
    var_28 = module_0.UnsupportedSettings(var_27)
    var_29 = str(var_28)
    var_30 = 'test_opt'
    var_31 = 'test_val'
    var_32 = 'source.py'
    var_33 = 'count'
    var_34 = 42
    var_35 = 'config'
    var_36 = 'flag'
    var_37 = True
    var_38 = 'items'
    var_39 = 'a'
    var_40 = 'b'
    var_41 = [var_39, var_40]
    var_42 = 'env'



# Parsed testcases at query #9
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = '/path/to/file.py'
    var_1 = module_0.IntroducedSyntaxErrors(var_0)
    var_2 = str(var_1)

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'module/submodule/file.py'
    var_2 = ''
    var_3 = [var_0, var_1, var_2]

import isort.exceptions as module_0

def test_case_0():
    var_0 = '/path/to/file.py'
    var_1 = module_0.IntroducedSyntaxErrors(var_0)
    var_2 = type(var_1)
    var_3 = var_1.__dict__
    var_4 = 0
    var_5 = 1



# Parsed testcases at query #10
#--------------------------


import isort.exceptions as module_0

def test_case_0():
    var_0 = 'Test UnsupportedSettings exception initialization and formatting'
    var_1 = 'invalid_option'
    var_2 = 'value'
    var_3 = 'source'
    var_4 = 'some_value'
    var_5 = 'config.ini'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.UnsupportedSettings(var_7)
    var_9 = str(var_8)
    var_10 = str(var_8)
    var_11 = str(var_8)
    var_12 = str(var_8)
    var_13 = 'bad_option1'
    var_14 = 'bad_option2'
    var_15 = 'bad_option3'
    var_16 = 'value1'
    var_17 = 'cli'
    var_18 = {var_2: var_16, var_3: var_17}
    var_19 = 'value2'
    var_20 = 'pyproject.toml'
    var_21 = {var_2: var_19, var_3: var_20}
    var_22 = 123
    var_23 = 'setup.cfg'
    var_24 = {var_2: var_22, var_3: var_23}
    var_25 = {var_13: var_18, var_14: var_21, var_15: var_24}
    var_26 = module_0.UnsupportedSettings(var_25)
    var_27 = str(var_26)
    var_28 = str(var_26)
    var_29 = str(var_26)
    var_30 = str(var_26)
    var_31 = str(var_26)
    var_32 = str(var_26)
    var_33 = {}
    var_34 = module_0.UnsupportedSettings(var_33)
    var_35 = str(var_34)
    var_36 = 'test_option'
    var_37 = 'test_value'
    var_38 = 'test_source'



