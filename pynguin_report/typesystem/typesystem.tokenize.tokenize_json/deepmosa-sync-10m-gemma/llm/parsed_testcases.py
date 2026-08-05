####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_json_string_scalar. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_list. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 123
    var_3 = var_1.string
    assert var_3 == '123'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '123.45'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, "two"]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, "two"]'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"bytes"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'bytes'
    var_3 = var_1.string
    assert var_3 == '"bytes"'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{\n  "a": 1\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.start.line_no
    assert var_5 == 2
    var_6 = var_4.start.column_no
    assert var_6 == 3



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 9/17 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 9/17 statements.
# Partially parsed test_tokenizing_json_object_error_no_quotes. Retrieved 9/18 statements.
# Partially parsed test_tokenizing_json_object_success. Retrieved 9/17 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = {}
    var_4 = '{}'
    var_5 = '{}'
    var_6 = 0
    var_7 = (var_5, var_6)
    var_8 = True

import re as module_0

def test_case_0():
    var_0 = '{"key":"value"}'
    var_1 = '\\s*'
    var_2 = module_0.compile(var_1)
    var_3 = ' \t\n\r'
    var_4 = {}
    var_5 = '{"key":"value"}'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = 'key'

import re as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = '\\s*'
    var_2 = module_0.compile(var_1)
    var_3 = ' \t\n\r'
    var_4 = {}
    var_5 = '{key: "value"}'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = True

import re as module_0

def test_case_0():
    var_0 = '{"a":"b"}'
    var_1 = '\\s*'
    var_2 = module_0.compile(var_1)
    var_3 = ' \t\n\r'
    var_4 = {}
    var_5 = '{"a":"b"}'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = True
    var_9 = 'a'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 12/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 12/15 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 12/15 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 12/16 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 12/15 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_char. Retrieved 13/17 statements.
# Partially parsed test_make_scanner_logic_flow. Retrieved 12/15 statements.


def test_case_0():
    var_0 = '"hello"'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = 'hello'
    var_6 = 5
    var_7 = lambda s, i, st: (var_5, i + var_6)
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}

def test_case_0():
    var_0 = 'null'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = ''
    var_6 = (var_5, var_2)
    var_7 = lambda s, i, st: var_6
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}

def test_case_0():
    var_0 = 'true'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = ''
    var_6 = (var_5, var_2)
    var_7 = lambda s, i, st: var_6
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}

def test_case_0():
    var_0 = '123'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = ''
    var_6 = (var_5, var_2)
    var_7 = lambda s, i, st: var_6
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}

def test_case_0():
    var_0 = '123.45'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = ''
    var_6 = (var_5, var_2)
    var_7 = lambda s, i, st: var_6
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}

def test_case_0():
    var_0 = '?'
    var_1 = []
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x, y: var_3
    var_5 = ''
    var_6 = (var_5, var_2)
    var_7 = lambda s, i, st: var_6
    var_8 = True
    var_9 = lambda x: float(x)
    var_10 = lambda x: int(x)
    var_11 = {}
    var_12 = 0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda s, i, st: var_5
    var_7 = True
    var_8 = lambda x: float(x)
    var_9 = lambda x: int(x)
    var_10 = {}
    var_11 = 'true'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 17/25 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 9/19 statements.
# Partially parsed test_tokenizing_json_object_empty_logic. Retrieved 14/19 statements.


import re as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' '
    var_8 = True
    var_9 = 'val'
    var_10 = 2
    var_11 = 4
    var_12 = module_1.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = lambda s, e: var_14
    var_16 = var_6.match

import re as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = {}
    var_2 = '{"a":1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' '
    var_8 = True

import re as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.tokenize.tokenize_json as module_2

def test_case_0():
    var_0 = '{a:1}'
    var_1 = {}
    var_2 = '{a:1}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' '
    var_8 = True
    var_9 = 3
    var_10 = module_1.ScalarToken(var_8, var_9, var_9, var_0)
    var_11 = 4
    var_12 = (var_10, var_11)
    var_13 = lambda s, e: var_12
    var_14 = var_6.match
    var_15 = module_2._TokenizingJSONObject(var_4, var_8, var_13, var_1, var_0, var_14, var_7)

import re as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' '
    var_8 = True
    var_9 = ''
    var_10 = module_1.ScalarToken(var_8, var_3, var_3, var_9)
    var_11 = (var_10, var_8)
    var_12 = lambda s, e: var_11
    var_13 = var_6.match



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0
    var_3 = var_1[var_2]
    assert var_3 == 'n'
    var_4 = 'n'
    var_5 = var_3 == var_4
    var_6 = 4
    var_7 = var_2 + var_6
    var_8 = var_1[var_2:var_7]
    var_9 = 'null'
    var_10 = var_8 == var_9
    var_11 = var_5 and var_10
    assert var_11 is True
    var_12 = var_1[var_2:var_2 + 4]
    assert var_12 == 'null'
    var_13 = None
    var_14 = var_2 + var_6
    assert var_14 == 4
    var_15 = 1
    var_16 = var_14 - var_15
    var_17 = module_0.ScalarToken(var_13, var_2, var_16, var_0)
    var_18 = var_17.value
    assert var_18 is None
    var_19 = var_17.string
    assert var_19 == 'null'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenizing_json_object_success. Retrieved 32/40 statements.
# Partially parsed test_scalar_token_hash_and_value. Retrieved 6/25 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{}'
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = None
    var_6 = lambda s, end: (var_5, end)

import builtins as module_0

def test_case_0():
    var_0 = 'fake'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'match'
    var_4 = 'M'
    var_5 = ()
    var_6 = 'end'
    var_7 = lambda s, i: type(var_4, var_5, {var_6: lambda : i})()
    var_8 = {var_3: var_7}
    var_9 = [var_1, var_2, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'Token'
    var_13 = ()
    var_14 = '_get_value'
    var_15 = '_start_index'
    var_16 = '_end_index'
    var_17 = '_content'
    var_18 = 'string'
    var_19 = '__eq__'
    var_20 = 'key'
    var_21 = lambda : var_20
    var_22 = lambda : var_20
    var_23 = 0
    var_24 = 2
    var_25 = '{"key":1}'
    var_26 = lambda : var_20
    var_27 = '"key"'
    var_28 = lambda : var_27
    var_29 = True
    var_30 = lambda s, o: var_29
    var_31 = {var_14: var_21, var_14: var_22, var_15: var_23, var_16: var_24, var_17: var_25, var_14: var_26, var_18: var_28, var_19: var_30}
    var_32 = [var_12, var_13, var_31]
    var_33 = {}
    var_34 = module_0.type(*var_32, **var_33)
    var_35 = var_34()

def test_case_0():
    var_0 = 123
    var_1 = 0
    var_2 = 2
    var_3 = '123'
    var_4 = 456
    var_5 = '456'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_tokenizing_json_object_colon_separator_fast_path_actual. Retrieved 7/11 statements.


import re as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '\\s+'
    var_2 = module_0.compile(var_1)
    var_3 = ' '
    var_4 = {}
    var_5 = True
    var_6 = '{"key" : "value"}'
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = '{"key" : "value"}'
    var_10 = (var_6, var_7)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = '{"key" : "value"}'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenizing_json_object_index_error_on_nextchar_check. Retrieved 7/27 statements.


def test_case_0():
    var_0 = ' '
    var_1 = {}
    var_2 = ' '
    var_3 = ' '
    var_4 = 0
    var_5 = ' '
    var_6 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenizing_json_object_skips_whitespace_before_colon. Retrieved 19/27 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = '{"key" : 1}'
    var_4 = '{"key" : 1}'
    var_5 = 5
    var_6 = {}
    var_7 = '{"key" : 1}'
    var_8 = 5
    var_9 = 1
    var_10 = var_8 + var_9
    var_11 = var_7[var_8:var_10]
    var_12 = ':'
    var_13 = var_11 != var_12
    assert var_13 is True
    var_14 = module_0.compile(var_0)
    var_15 = module_0.search(var_7, var_8)
    var_16 = var_8 + var_9
    var_17 = var_7[var_8:var_16]
    var_18 = var_17 != var_12
    assert var_18 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 16/17 statements.
# Partially parsed test_tokenizing_json_object_value_lookup. Retrieved 16/28 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '{"": 1}'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = '}'
    var_9 = (var_8, var_3)
    var_10 = True
    var_11 = None
    var_12 = (var_11, var_3)
    var_13 = lambda x, y: var_12
    var_14 = {}
    var_15 = lambda x: var_11

def test_case_0():
    pass

def test_case_0():
    var_0 = '{key: 1}'
    var_1 = {}
    var_2 = '{key: 1}'
    var_3 = 1
    var_4 = (var_2, var_3)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = 3
    var_3 = '"key"'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 123
    var_6 = 5
    var_7 = 7
    var_8 = '123'
    var_9 = module_0.ScalarToken(var_5, var_6, var_7, var_8)
    var_10 = None
    var_11 = 10
    var_12 = '{"a": 123}'
    var_13 = 'a'
    var_14 = [var_13]
    var_15 = [var_13]

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 2
    var_3 = '10'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = var_4.value
    assert var_5 == 10
    var_6 = var_4.string
    assert var_6 == '10'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = '1'
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_5 = 2
    var_6 = '2'
    var_7 = module_0.ScalarToken(var_5, var_1, var_1, var_6)
    var_8 = bool(var_3 == var_4)
    assert var_8 is True
    var_9 = bool(var_3 != var_7)
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_scan_once_null_branch_not_taken. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'name'
    var_2 = 0
    var_3 = var_1[var_2]
    var_4 = 'n'
    var_5 = var_3 == var_4
    var_6 = 4
    var_7 = var_2 + var_6
    var_8 = var_1[var_2:var_7]
    var_9 = 'null'
    var_10 = var_8 == var_9
    var_11 = var_5 and var_10
    assert var_11 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 26/39 statements.
# Partially parsed test_tokenizing_json_object_simple_pair. Retrieved 21/34 statements.


import builtins as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = '{}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = None
    var_7 = lambda s, end: (var_6, end)
    var_8 = 'obj'
    var_9 = ()
    var_10 = 'match'
    var_11 = 'm'
    var_12 = ()
    var_13 = 'end'
    var_14 = lambda s, e: type(var_11, var_12, {var_13: lambda : e})()
    var_15 = {var_10: var_14}
    var_16 = [var_8, var_9, var_15]
    var_17 = {}
    var_18 = module_0.type(*var_16, **var_17)
    var_19 = ' \t\n\r'
    var_20 = 'scanstring'
    var_21 = 'typesystem.tokenize.tokenize_json'
    var_22 = '"key"'
    var_23 = '{'
    var_24 = (var_23, var_2)
    var_25 = True
    var_26 = lambda s, end: (var_6, end)
    var_27 = {}

import builtins as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = 'obj'
    var_1 = ()
    var_2 = 'match'
    var_3 = 'm'
    var_4 = ()
    var_5 = 'end'
    var_6 = lambda s, e: type(var_3, var_4, {var_5: lambda : e})()
    var_7 = {var_2: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = 1
    var_12 = 4
    var_13 = '{"a":1}'
    var_14 = module_1.ScalarToken(var_11, var_12, var_12, var_13)
    var_15 = 5
    var_16 = (var_14, var_15)
    var_17 = '{"a":1}'
    var_18 = 0
    var_19 = (var_13, var_18)
    var_20 = {}
    var_21 = True
    var_22 = ' '
    var_23 = 'a'



# Parsed testcases at query #13
#--------------------------




import re as module_0

def test_case_0():
    var_0 = '{"key": "value", "next": "value"}'
    var_1 = 1
    var_2 = True
    var_3 = {}
    var_4 = '\\s*'
    var_5 = module_0.compile(var_4)
    var_6 = ' '

def test_case_0():
    var_0 = '{"a":"b",}'
    var_1 = 1
    var_2 = {}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_null_token. Retrieved 3/21 statements.
# Partially parsed test_make_scanner_true_token. Retrieved 3/21 statements.
# Partially parsed test_make_scanner_string_token. Retrieved 3/21 statements.


def test_case_0():
    var_0 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_1 = 'true'
    var_2 = 0

def test_case_0():
    var_0 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_1 = '"hello"'
    var_2 = 0

def test_case_0():
    pass



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_scans_true_token. Retrieved 11/18 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 11/18 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 11/18 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 11/18 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_char. Retrieved 11/18 statements.


import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = lambda self, x, y: ([], 0)
    var_3 = lambda self, s, i, st: ('str', 5)
    var_4 = lambda self, x: 1.0
    var_5 = lambda self, x: 1
    var_6 = '(\\d+)(\\.(\\d+))?(e([+-]?\\d+))?'
    var_7 = module_0.compile(var_6)
    var_8 = 'true'
    var_9 = 'true'
    var_10 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = lambda self, x, y: ([], 0)
    var_3 = lambda self, s, i, st: ('hello', i + 5)
    var_4 = lambda self, x: 1.0
    var_5 = lambda self, x: 1
    var_6 = '(\\d+)(\\.(\\d+))?(e([+-]?\\d+))?'
    var_7 = module_0.compile(var_6)
    var_8 = '"hello"'
    var_9 = '"hello"'
    var_10 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = lambda self, x, y: ([], 0)
    var_3 = lambda self, s, i, st: ('str', 5)
    var_4 = lambda self, x: 1.0
    var_5 = lambda self, x: 123
    var_6 = '(\\d+)(\\.(\\d+))?(e([+-]?\\d+))?'
    var_7 = module_0.compile(var_6)
    var_8 = '123'
    var_9 = '123'
    var_10 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = lambda self, x, y: ([], 0)
    var_3 = lambda self, s, i, st: ('str', 5)
    var_4 = lambda self, x: 1.0
    var_5 = lambda self, x: 1
    var_6 = '(\\d+)(\\.(\\d+))?(e([+-]?\\d+))?'
    var_7 = module_0.compile(var_6)
    var_8 = 'null'
    var_9 = 'null'
    var_10 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = lambda self, x, y: ([], 0)
    var_3 = lambda self, s, i, st: ('str', 5)
    var_4 = lambda self, x: 1.0
    var_5 = lambda self, x: 1
    var_6 = '(\\d+)(\\.(\\d+))?(e([+-]?\\d+))?'
    var_7 = module_0.compile(var_6)
    var_8 = '!'
    var_9 = '!'
    var_10 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_json_string_scalar. Retrieved 7/9 statements.
# Partially parsed test_tokenize_json_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 6/7 statements.
# Partially parsed test_tokenize_json_list. Retrieved 6/7 statements.
# Partially parsed test_tokenize_json_bytes. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 8
    var_10 = 7
    var_11 = var_1.end

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = 'key'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 'value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, "two"]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two'])
    assert var_3 is True
    var_4 = 1
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 'two'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"byte_string"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'byte_string'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"unclosed": "quote}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 13/16 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 11/23 statements.
# Partially parsed test_tokenizing_json_object_error_missing_quote. Retrieved 15/20 statements.
# Partially parsed test_tokenizing_json_object_error_missing_colon. Retrieved 10/21 statements.


import typesystem.tokenize.tokens as module_0
import re as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = '{}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = None
    var_7 = module_0.ScalarToken(var_6, var_2, var_2, var_0)
    var_8 = lambda s, end: (var_7, end)
    var_9 = '\\s*'
    var_10 = module_1.compile(var_9)
    var_11 = ' '
    var_12 = var_10.match

import re as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = '\\s*'
    var_7 = module_0.compile(var_6)
    var_8 = ' '
    var_9 = var_7.match
    var_10 = 'key'

import typesystem.tokenize.tokens as module_0
import re as module_1
import typesystem.tokenize.tokenize_json as module_2

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = '{key: "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = None
    var_7 = module_0.ScalarToken(var_6, var_2, var_2, var_0)
    var_8 = lambda s, end: (var_7, end)
    var_9 = '\\s*'
    var_10 = module_1.compile(var_9)
    var_11 = ' '
    var_12 = 'bad'
    var_13 = var_10.match
    var_14 = module_2._TokenizingJSONObject(var_3, var_4, var_8, var_5, var_0, var_13, var_11)
    var_15 = 'Expecting property name enclosed in double quotes'

import re as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = '{"key" "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = '\\s*'
    var_7 = module_0.compile(var_6)
    var_8 = ' '
    var_9 = var_7.match
    var_10 = "Expecting ':' delimiter"



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_scanner_scans_string. Retrieved 6/23 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 6/22 statements.
# Partially parsed test_make_scanner_scans_bool_true. Retrieved 6/22 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 6/22 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_empty. Retrieved 6/24 statements.


import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = '"hello"'
    var_5 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = 'null'
    var_5 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = 'true'
    var_5 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = '123.45'
    var_5 = 0

import re as module_0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '(-?\\d+)(?:\\.(\\d+))?(?:[eE]([+-]?\\d+))?'
    var_3 = module_0.compile(var_2)
    var_4 = ''
    var_5 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenizing_json_object_whitespace_after_value. Retrieved 24/79 statements.


import re as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = {}
    var_4 = '{"k": "v" }'
    var_5 = '{"k": "v" }'
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = (var_5, var_6)
    var_9 = True
    var_10 = 'v'
    var_11 = 5
    var_12 = module_1.ScalarToken(var_10, var_11, var_11, var_5)
    var_13 = 7
    var_14 = (var_12, var_13)
    var_15 = lambda s, e: var_14
    var_16 = {}
    var_17 = var_1.match
    var_18 = ' \t\n\r'
    var_19 = 'k'
    var_20 = 2
    var_21 = module_1.ScalarToken(var_19, var_9, var_20, var_5)
    var_22 = module_1.ScalarToken(var_10, var_11, var_11, var_5)
    var_23 = {var_21: var_22}



# Parsed testcases at query #6
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0
    var_3 = None
    var_4 = 4
    var_5 = var_2 + var_4
    var_6 = 1
    var_7 = var_5 - var_6
    var_8 = module_0.ScalarToken(var_3, var_2, var_7, var_0)
    var_9 = (var_8, var_5)
    var_10 = var_9[0].value
    assert var_10 is None
    var_11 = var_9[0].start
    assert var_11 == 0
    var_12 = var_9[0].end
    assert var_12 == 3
    var_13 = var_9[0].string
    assert var_13 == 'null'
    var_14 = var_9[1]
    assert var_14 == 4



# Parsed testcases at query #7
#--------------------------




import re as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \n\t\r'
    var_3 = '{"key": "value"}'
    var_4 = '{"key": "value"}'
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = {}
    var_8 = 'value'
    var_9 = 8
    var_10 = 13
    var_11 = module_1.ScalarToken(var_8, var_9, var_10, var_3)
    var_12 = 14
    var_13 = (var_11, var_12)
    var_14 = lambda s, e: var_13
    var_15 = 'key'
    var_16 = 4
    var_17 = module_1.ScalarToken(var_15, var_5, var_16, var_3)
    var_18 = 5
    var_19 = (var_17, var_18)
    var_20 = lambda s, e, strict: var_19
    var_21 = var_5 + var_5
    var_22 = var_4[var_5:var_21]
    var_23 = bool(var_22 != '}')
    assert var_23 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_scanner_null_token_detection. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_line_46_evaluates_to_true. Retrieved 9/10 statements.


def test_case_0():
    var_0 = '"key":  ,"next":"val"}'
    var_1 = '"key"'
    var_2 = 6
    var_3 = (var_1, var_2)
    var_4 = {}
    var_5 = '"key":  ,"next":"val"}'
    var_6 = (var_1, var_2)
    var_7 = {}
    var_8 = True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_TokenizingJSONObject_skips_colon_separator_with_whitespace. Retrieved 13/25 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \n\r\t'
    var_3 = '{"key": true}'
    var_4 = {}
    var_5 = True
    var_6 = '{"key"'
    var_7 = 7
    var_8 = (var_6, var_7)
    var_9 = '{"key" : true}'
    var_10 = 6
    var_11 = (var_6, var_10)
    var_12 = var_1.match



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenizing_json_object_nextchar_not_quote. Retrieved 15/22 statements.


import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \t\n\r'
    var_3 = '{ '
    var_4 = 1
    var_5 = '{ '
    var_6 = {}
    var_7 = True
    var_8 = (var_3, var_4)
    var_9 = '{  }'
    var_10 = 1
    var_11 = '{  }'
    var_12 = {}
    var_13 = (var_9, var_10)
    var_14 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenizing_json_object_comma_delimiter_success. Retrieved 14/31 statements.


import re as module_0

def test_case_0():
    var_0 = '{"a":1,"b":2}'
    var_1 = '\\s*'
    var_2 = module_0.compile(var_1)
    var_3 = var_2.match
    var_4 = ' \t\n\r'
    var_5 = {}
    var_6 = 1
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = {}
    var_10 = module_0.compile(var_1)
    var_11 = var_10.match
    var_12 = ' \t\n\r'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 10/14 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 11/20 statements.
# Partially parsed test_tokenizing_json_object_error_no_quote. Retrieved 10/16 statements.
# Partially parsed test_tokenizing_json_object_error_no_colon. Retrieved 10/18 statements.
# Partially parsed test_tokenizing_json_object_error_no_comma. Retrieved 10/22 statements.


import re as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{}'
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' \n\r\t'
    var_8 = True
    var_9 = var_6.match

import re as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' \n\r\t'
    var_8 = True
    var_9 = var_6.match
    var_10 = 'key'
    var_11 = 'key'

import re as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{key: "value"}'
    var_2 = '{key: "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' \n\r\t'
    var_8 = True
    var_9 = var_6.match
    var_10 = 'Expecting property name enclosed in double quotes'

import re as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{"key" "value"}'
    var_2 = '{"key" "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' \n\r\t'
    var_8 = True
    var_9 = var_6.match
    var_10 = "Expecting ':' delimiter"

import re as module_0

def test_case_0():
    var_0 = {}
    var_1 = '{"a": 1 "b": 2}'
    var_2 = '{"a": 1 "b": 2}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = '\\s*'
    var_6 = module_0.compile(var_5)
    var_7 = ' \n\r\t'
    var_8 = True
    var_9 = var_6.match
    var_10 = "Expecting ',' delimiter"



# Parsed testcases at query #14
#--------------------------




import re as module_0

def test_case_0():
    var_0 = '\\s*'
    var_1 = module_0.compile(var_0)
    var_2 = ' \n\t\r'
    var_3 = '{"key": "value" '
    var_4 = '{"key": "value" '
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = {}
    var_8 = '{"k": "v" '
    var_9 = (var_8, var_5)
    var_10 = {}
    var_11 = '{"k": "v" '
    var_12 = True
    var_13 = 'v'
    var_14 = lambda s, end: (ScalarToken(var_13, end, end, var_11), end + var_5)
    var_15 = 'k'
    var_16 = lambda s, end, strict: (ScalarToken(var_15, end, end, var_11), end + var_5)
    var_17 = lambda s, end: re.search(var_0, s[end:])



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_TokenizingJSONObject_nextchar_is_whitespace. Retrieved 15/95 statements.


def test_case_0():
    var_0 = ' \t\n\r'
    var_1 = '{"a":1, "b":2}'
    var_2 = '{"a"'
    var_3 = 4
    var_4 = (var_2, var_3)
    var_5 = {}
    var_6 = True
    var_7 = '{"a":1, "b":2}'
    var_8 = '{"a":1 , "b":2}'
    var_9 = '{"a":1 , "b":2}'
    var_10 = {}
    var_11 = True
    var_12 = ' \t\n\r'
    var_13 = (var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_json_string_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_number_int_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_number_float_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_boolean_true_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_boolean_false_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_null_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_list_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_dict_success. Retrieved 2/5 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 123

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 123.45)
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[1, "a"]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'a'])
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"byte"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'byte'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": unquoted}'
    var_1 = module_0.tokenize_json(var_0)



