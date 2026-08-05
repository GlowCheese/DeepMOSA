####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_string. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_null. Retrieved 2/5 statements.
# Partially parsed test_make_scanner_true. Retrieved 2/5 statements.
# Partially parsed test_make_scanner_false. Retrieved 2/5 statements.
# Partially parsed test_make_scanner_int. Retrieved 2/5 statements.
# Partially parsed test_make_scanner_float. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '"hello"'
    var_1 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 10/12 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 18/33 statements.
# Partially parsed test_tokenizing_json_object_error_missing_colon. Retrieved 16/30 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = module_0.ScalarToken(var_5, var_3, var_3, var_0)
    var_7 = lambda s, end: (var_6, end)
    var_8 = (var_0, var_3)
    var_9 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = {}
    var_2 = 'a'
    var_3 = 1
    var_4 = module_0.ScalarToken(var_2, var_3, var_3, var_0)
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = 'obj'
    var_8 = 'match'
    var_9 = 'end'
    var_10 = 2
    var_11 = lambda : var_10
    var_12 = {var_9: var_11}
    var_13 = '{"a":1}'
    var_14 = 0
    var_15 = (var_13, var_14)
    var_16 = True
    var_17 = module_0.ScalarToken(var_2, var_16, var_16, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{a:1}'
    var_1 = {}
    var_2 = '{a:1}'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, e: (var_6, e)
    var_8 = module_0._TokenizingJSONObject(var_4, var_5, var_7, var_1, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a" 1}'
    var_1 = {}
    var_2 = 'a'
    var_3 = 1
    var_4 = module_0.ScalarToken(var_2, var_3, var_3, var_0)
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = 'obj'
    var_8 = 'match'
    var_9 = 'end'
    var_10 = lambda : var_5
    var_11 = {var_9: var_10}
    var_12 = '{"a" 1}'
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenizing_json_object_empty_immediately. Retrieved 19/21 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = '{}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'val'
    var_6 = 2
    var_7 = 4
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_1)
    var_9 = 5
    var_10 = (var_8, var_9)
    var_11 = lambda s, end: var_10
    var_12 = {}
    var_13 = '{}'
    var_14 = 'Match'
    var_15 = ()
    var_16 = 'end'
    var_17 = lambda s, pos: type(var_8, var_9, {var_10: lambda : pos})()
    var_18 = ' \n\t'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenizing_json_object_colon_after_whitespace. Retrieved 14/29 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key"'
    var_2 = 6
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = 'value'
    var_6 = 9
    var_7 = 14
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 15
    var_10 = (var_8, var_9)
    var_11 = lambda s, e: var_10
    var_12 = '{"key" : "value"}'
    var_13 = 7



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_scanner_null_predicate_false. Retrieved 7/25 statements.


def test_case_0():
    var_0 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_1 = 'notnull'
    var_2 = 'n'
    var_3 = 0
    var_4 = 'Not a string'
    var_5 = 'n'
    var_6 = 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenizing_json_object_nextchar_is_comma. Retrieved 10/27 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' '
    var_3 = '{"k":"v","k2":"v2"}'
    var_4 = {}
    var_5 = '{"k":"v", "k2":"v2"}'
    var_6 = 0
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = var_1.match

def test_case_0():
    var_0 = '{"k":"v",}'
    var_1 = '{"k":"name"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = '{"k":"v",}'
    var_5 = {}
    var_6 = '{"k":"v",}'
    var_7 = 0
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = '{"k":"v",}'
    var_1 = '{"k":"v",}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = 7
    var_6 = var_0[var_5]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 15/30 statements.
# Partially parsed test_tokenizing_json_object_error_no_quote. Retrieved 8/19 statements.


import re as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = ''
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = '{}'
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = {}
    var_8 = '\\s*'
    var_9 = module_0.compile(var_8)
    var_10 = ' '
    var_11 = '}'
    var_12 = (var_11, var_2)
    var_13 = True
    var_14 = {}

import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' '
    var_3 = '{'
    var_4 = 0
    var_5 = (var_3, var_4)
    var_6 = True
    var_7 = {}

import re as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = '"a":1}'
    var_2 = 0
    var_3 = {}
    var_4 = '\\s*'
    var_5 = module_0.compile(var_4)
    var_6 = ' '
    var_7 = '"a":1}'
    var_8 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_scan_once_null_token. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenizing_json_object_skips_index_error_on_whitespace_end. Retrieved 10/18 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s+'
    var_1 = module_0.compile(var_0)
    var_2 = ' \n\t\r'
    var_3 = {}
    var_4 = '{"a": '
    var_5 = '{"a": '
    var_6 = 5
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = var_1.match



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_array. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_object. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_invalid_syntax_error. Retrieved 2/6 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 4
    var_6 = 3
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = 7
    var_6 = 6
    var_7 = module_1.Position(var_2, var_5, var_6)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, "a"]'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"byte"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": }'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, {"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = 'b'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 1/16 statements.


def test_case_0():
    var_0 = '{"a": 1}'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_json_success. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenizing_json_object_index_error_not_raised. Retrieved 13/27 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \n\t\r'
    var_3 = '{"key":v}'
    var_4 = {}
    var_5 = '{"key"'
    var_6 = 7
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = '{"key":v}'
    var_10 = (var_9, var_6)
    var_11 = True
    var_12 = var_1.match



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 4/18 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 5/22 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 5/22 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_eof. Retrieved 4/18 statements.


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '"hello"'
    var_3 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(\\d+)(?:\\.(\\d+))?(?:e([+-]?\\d+))?'
    var_3 = '123'
    var_4 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(\\d+)(?:\\.(\\d+))?(?:e([+-]?\\d+))?'
    var_3 = '1.23'
    var_4 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = ''
    var_3 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_json_success. Retrieved 2/4 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 8/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{}'
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True

def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{key: "value"}'
    var_2 = '{key: "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = module_0._TokenizingJSONObject(var_4, var_7, var_6, var_0, var_1)

def test_case_0():
    var_0 = {}
    var_1 = '{"a":"b"}'
    var_2 = '{"a":"b"}'
    var_3 = 0
    var_4 = (var_2, var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 4/12 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/9 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/9 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/9 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 4/14 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 4/14 statements.
# Partially parsed test_make_scanner_scans_list. Retrieved 6/14 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 11/20 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_eof. Retrieved 2/9 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '"hello"'
    var_1 = 'hello'
    var_2 = 7
    var_3 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

import re as module_0

def test_case_0():
    var_0 = '(-?(?P<integer>\\d+)(?P<frac>\\.\\d+)?(?P<exp>[eE][+-]?\\d+)?)'
    var_1 = module_0.compile(var_0)
    var_2 = '123'
    var_3 = 0

import re as module_0

def test_case_0():
    var_0 = '(-?(?P<integer>\\d+)(?P<frac>\\.\\d+)?(?P<exp>[eE][+-]?\\d+)?)'
    var_1 = module_0.compile(var_0)
    var_2 = '123.45'
    var_3 = 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '[1]'
    var_1 = 1
    var_2 = module_0.ScalarToken(var_1, var_1, var_1, var_0)
    var_3 = [var_2]
    var_4 = 3
    var_5 = 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = 'a'
    var_2 = 1
    var_3 = module_0.ScalarToken(var_1, var_2, var_2, var_0)
    var_4 = 4
    var_5 = module_0.ScalarToken(var_2, var_4, var_4, var_0)
    var_6 = {var_1: var_5}
    var_7 = 6
    var_8 = {var_1: var_5}
    var_9 = (var_8, var_7)
    var_10 = 0

def test_case_0():
    var_0 = ''
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 'old'
    var_2 = 'data'
    var_3 = 0



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 2/17 statements.
# Partially parsed test_make_scanner_scans_bool_true. Retrieved 1/13 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 3/33 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/19 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/19 statements.


def test_case_0():
    var_0 = '"hello"'
    var_1 = 0

def test_case_0():
    var_0 = 'true'

def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 10/15 statements.
# Partially parsed test_tokenizing_json_object_with_pair. Retrieved 7/16 statements.


import re as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '\\s*'
    var_3 = module_0.compile(var_2)
    var_4 = ' \t\n\r'
    var_5 = '{}'
    var_6 = 0
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = var_3.match

import re as module_0

def test_case_0():
    var_0 = '{"key":123}'
    var_1 = {}
    var_2 = '\\s*'
    var_3 = module_0.compile(var_2)
    var_4 = ' \t\n\r'
    var_5 = 'scanstring'
    var_6 = None

import re as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = {}
    var_2 = '\\s*'
    var_3 = module_0.compile(var_2)
    var_4 = ' \t\n\r'
    var_5 = '{key: 123}'
    var_6 = 0
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = None
    var_10 = lambda s, e: (var_9, e)
    var_11 = var_3.match
    var_12 = module_1._TokenizingJSONObject(var_7, var_8, var_10, var_1, var_0, var_11, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_scan_once_null_detection. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_scan_once_identifies_string_token. Retrieved 3/21 statements.


def test_case_0():
    var_0 = '"abc"'
    var_1 = '"abc"'
    var_2 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 4/11 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '"hello"'
    var_1 = 'hello'
    var_2 = 7
    var_3 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 0

import re as module_0

def test_case_0():
    var_0 = '(-?(?:\\d+)(?:\\.(\\d+)?(?:[eE]([+-]?\\d+))?)?)'
    var_1 = module_0.compile(var_0)
    var_2 = '123'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenizing_json_object_not_empty_brace. Retrieved 14/28 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = '{"key": "value"}'
    var_4 = 1
    var_5 = 1
    var_6 = (var_3, var_5)
    var_7 = 'value'
    var_8 = 7
    var_9 = 12
    var_10 = {}
    var_11 = var_3
    var_12 = True
    var_13 = var_1.match



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_scanner_string. Retrieved 7/22 statements.
# Partially parsed test_make_scanner_null. Retrieved 7/22 statements.
# Partially parsed test_make_scanner_true. Retrieved 7/22 statements.
# Partially parsed test_make_scanner_number_int. Retrieved 7/22 statements.
# Partially parsed test_make_scanner_number_float. Retrieved 7/22 statements.


import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = '"hello"'
    var_5 = '"hello"'
    var_6 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = 'null'
    var_5 = 'null'
    var_6 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = 'true'
    var_5 = 'true'
    var_6 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = '123'
    var_5 = '123'
    var_6 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = '1.23'
    var_5 = '1.23'
    var_6 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_scanner_string. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_null. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_true. Retrieved 5/22 statements.
# Partially parsed test_make_scanner_number_int. Retrieved 8/27 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 5/19 statements.


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '"val"'
    var_3 = '"val"'
    var_4 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = 'null'
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = 'true'
    var_3 = 'true'
    var_4 = 0

import re as module_0

def test_case_0():
    var_0 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_1 = module_0.compile(var_0)
    var_2 = False
    var_3 = {}
    var_4 = 'typesystem.tokenize.tokenize_json'
    var_5 = '123'
    var_6 = '123'
    var_7 = 0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = '!'
    var_3 = '!'
    var_4 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_json_empty_string. Retrieved 3/6 statements.
# Partially parsed test_tokenize_json_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_number_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_number_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_list. Retrieved 10/11 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 9/10 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = str(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, "a"]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 1
    var_6 = 1
    var_7 = [var_6]
    var_8 = var_1.lookup(var_7)
    var_9 = var_8.value
    assert var_9 == 'a'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = [var_2]
    var_7 = var_1.lookup_key(var_6)
    var_8 = var_7.value
    assert var_8 == 'value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"bytes"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{\n  "a": 1\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"unclosed": "val"'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenizing_json_object_nextchar_is_comma. Retrieved 12/21 statements.


import re as module_0

def test_case_0():
    var_0 = '{"key": "value", "next": "val"}'
    var_1 = '{"'
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = '{"key": "value", "next": "val"}'
    var_8 = (var_7, var_2)
    var_9 = True
    var_10 = var_6.match
    var_11 = ' \t\n\r'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_scan_once_null_predicate_false. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 'nothin'
    var_1 = 0
    var_2 = var_0[var_1]
    var_3 = 'n'
    var_4 = var_2 == var_3
    var_5 = 4
    var_6 = var_1 + var_5
    var_7 = var_0[var_1:var_6]
    var_8 = 'null'
    var_9 = var_7 == var_8
    var_10 = var_4 and var_9
    assert var_10 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 11/13 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 15/25 statements.
# Partially parsed test_tokenizing_json_object_error_missing_colon. Retrieved 19/26 statements.


import re as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = ' \n\r\t'
    var_9 = module_0.compile(var_8)
    var_10 = var_9.match

import typesystem.tokenize.tokens as module_0
import re as module_1

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = {}
    var_2 = '{"a":1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = 1
    var_7 = 2
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 4
    var_10 = module_0.ScalarToken(var_6, var_9, var_9, var_0)
    var_11 = True
    var_12 = ' \n\r\t'
    var_13 = module_1.compile(var_12)
    var_14 = var_13.match

import re as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{a:1}'
    var_1 = {}
    var_2 = '{a:1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = ' \n\r\t'
    var_9 = module_0.compile(var_8)
    var_10 = var_9.match
    var_11 = module_1._TokenizingJSONObject(var_4, var_7, var_6, var_1, var_0, var_10, var_8)

import re as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.tokenize_json as module_2

def test_case_0():
    var_0 = '{"a" 1}'
    var_1 = {}
    var_2 = '{"a" 1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = ' \n\r\t'
    var_7 = module_0.compile(var_6)
    var_8 = var_7.match
    var_9 = 'a'
    var_10 = 1
    var_11 = 2
    var_12 = module_1.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 5
    var_14 = module_1.ScalarToken(var_10, var_13, var_13, var_0)
    var_15 = 3
    var_16 = (var_12, var_15)
    var_17 = (var_14, var_13)
    var_18 = module_2._TokenizingJSONObject(var_4, var_5, var_2, var_1, var_0, var_8, var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenizing_json_object_index_error_at_end_of_string. Retrieved 8/24 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = '{"key":true'
    var_4 = {}
    var_5 = '{"key":true'
    var_6 = 1
    var_7 = '"'
    assert var_7 == ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenizing_json_object_colon_after_whitespace. Retrieved 9/30 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = ' \t\n\r'
    var_7 = '{"key" : "value"}'
    var_8 = 6



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 10/11 statements.
# Partially parsed test_tokenizing_json_object_with_whitespace. Retrieved 9/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = None
    var_9 = ''

def test_case_0():
    var_0 = '{  }'
    var_1 = {}
    var_2 = '{  }'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = ' '

def test_case_0():
    pass

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: 1}'
    var_1 = {}
    var_2 = '{key: 1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)
    var_7 = True
    var_8 = module_0._TokenizingJSONObject(var_4, var_7, var_6, var_1, var_0)

def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_scan_once_null_detection. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0
    var_3 = var_1[var_2]
    assert var_3 == 'n'
    var_4 = None
    var_5 = 4
    var_6 = var_2 + var_5
    var_7 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_scan_once_handles_null. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenizing_json_object_nextchar_is_comma. Retrieved 8/28 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' '
    var_3 = '{"a":1,"b":2}'
    var_4 = '{"a":1,"b":2}'
    var_5 = 0
    var_6 = (var_4, var_5)
    var_7 = {}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenizing_json_object_index_error_at_line_61. Retrieved 10/65 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' '
    var_3 = '{"a":1'
    var_4 = '{"a":'
    var_5 = 5
    var_6 = (var_4, var_5)
    var_7 = {}
    var_8 = (var_4, var_5)
    var_9 = True



