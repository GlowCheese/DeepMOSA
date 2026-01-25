####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}

def test_case_0():
    var_0 = '{"key":"value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1":"value1","key2":"value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":"value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1":"value1" "key2":"value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = lambda s, e: (ScalarToken(var_4, var_1, var_1, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 23/31 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 25/34 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 20/27 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = {}
    var_17 = True
    var_18 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = '123.45'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_2)
    var_6 = [var_3, var_5]
    var_7 = 'MockContext'
    var_8 = ()
    var_9 = 'parse_array'
    var_10 = 'parse_string'
    var_11 = 'parse_float'
    var_12 = 'parse_int'
    var_13 = 'memo'
    var_14 = 'strict'
    var_15 = 3
    var_16 = (var_6, var_15)
    var_17 = lambda x, y: var_16
    var_18 = (var_2, var_1)
    var_19 = lambda x, y, z: var_18
    var_20 = {}
    var_21 = True
    var_22 = '[1, 2]'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = 'value'
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_2)
    var_6 = {var_3: var_5}
    var_7 = 'MockContext'
    var_8 = ()
    var_9 = 'parse_array'
    var_10 = 'parse_string'
    var_11 = 'parse_float'
    var_12 = 'parse_int'
    var_13 = 'memo'
    var_14 = 'strict'
    var_15 = []
    var_16 = (var_15, var_1)
    var_17 = lambda x, y: var_16
    var_18 = (var_2, var_1)
    var_19 = lambda x, y, z: var_18
    var_20 = {}
    var_21 = True
    var_22 = 7
    var_23 = (var_6, var_22)
    var_24 = '{"key": "value"}'

def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'MockContext'
    var_4 = ()
    var_5 = 'parse_array'
    var_6 = 'parse_string'
    var_7 = 'parse_float'
    var_8 = 'parse_int'
    var_9 = 'memo'
    var_10 = 'strict'
    var_11 = []
    var_12 = 0
    var_13 = (var_11, var_12)
    var_14 = lambda x, y: var_13
    var_15 = ''
    var_16 = (var_15, var_12)
    var_17 = lambda x, y, z: var_16
    var_18 = True
    var_19 = 'null'
    var_20 = bool(var_2 == {})
    assert var_20 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_array_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/25 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = {}
    var_17 = True
    var_18 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = '123.45'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 2
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = {}
    var_17 = True
    var_18 = '[]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = True
    var_17 = '{}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = 'test'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #5
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = None
    var_2 = 8
    var_3 = 11
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = var_4.string
    assert var_5 == 'null'
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = 9
    var_9 = module_1.Position(var_7, var_8, var_2)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 12
    var_13 = module_1.Position(var_7, var_12, var_3)
    var_14 = var_4.end
    var_15 = bool(var_4.end == var_13)
    assert var_15 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 18/19 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = {}
    var_5 = None
    var_6 = lambda s, end: (ScalarToken(var_5, end, end, var_0), end)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = 'key'
    var_11 = len(var_0)
    var_12 = var_11 - var_2
    var_13 = len(var_0)
    var_14 = var_13 - var_2
    var_15 = module_0.ScalarToken(var_5, var_12, var_14, var_0)
    var_16 = {var_10: var_15}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '"test"'
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

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '[1]'
    var_1 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

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
    var_9 = 7
    var_10 = 6
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"list": [1, 2], "dict": {"a": 1}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'dict': {'a': 1}})
    assert var_3 is True
    var_4 = 'list'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'dict'
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    var_13 = bool(var_12 == {'a': 1})
    assert var_13 is True
    var_14 = 0
    var_15 = [var_4, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = 'a'
    var_19 = [var_9, var_18]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

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
    var_9 = 7
    var_10 = 6
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 9
    var_11 = 8
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 1
    var_18 = [var_5]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 2
    var_21 = 2
    var_22 = [var_21]
    var_23 = var_1.lookup(var_22)
    var_24 = var_23.value
    assert var_24 == 3

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = var_17.value
    assert var_18 == 'a'
    var_19 = [var_15]
    var_20 = var_1.lookup(var_19)
    var_21 = var_20.value
    assert var_21 == 1
    var_22 = 'b'
    var_23 = [var_22]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'b'
    var_26 = [var_22]
    var_27 = var_1.lookup(var_26)
    var_28 = var_27.value
    assert var_28 == 2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    var_17 = var_16.value
    assert var_17 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = ''
    var_5 = var_0[var_3]
    assert var_5 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 5/14 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = None
    var_3 = 10
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 12/14 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, end: (ScalarToken(var_4, end, end, s), end + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '  {  "key"  :  "value"  }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'inner'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 14
    var_8 = 15
    var_9 = lambda s, end: (ScalarToken(var_6, end, end + var_7, s), end + var_8)
    var_10 = {}
    var_11 = len(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 10/19 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 'value'
    var_2 = 7
    var_3 = (var_1, var_2)
    var_4 = 1
    var_5 = 8
    var_6 = module_0.Position(var_4, var_5, var_2)
    var_7 = 11
    var_8 = 10
    var_9 = module_0.Position(var_4, var_7, var_8)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_parse_object_is_not_tokenizing_json_object. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 20/28 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 20/26 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"test": "value"}'
    var_19 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '{"key": "value"}'
    var_18 = '{}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 2
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '[]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 'null'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 8/11 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = 1
    var_3 = 8
    var_4 = module_0.Position(var_2, var_3, var_1)
    var_5 = 11
    var_6 = 10
    var_7 = module_0.Position(var_2, var_5, var_6)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 18/23 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_raises_stop_iteration. Retrieved 19/25 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'test content'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 4
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"test": "value"}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 3
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"key": "value"}'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = 1
    var_9 = 0
    var_10 = ''
    var_11 = module_0.ScalarToken(var_8, var_9, var_9, var_10)
    var_12 = [var_11]
    var_13 = 2
    var_14 = lambda x, y: (var_12, y + var_13)
    var_15 = (var_10, var_8)
    var_16 = lambda x, y, z: var_15
    var_17 = False
    var_18 = {}
    var_19 = '[1]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = ''
    var_18 = 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 5
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"test": "value"}'
    var_18 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 3
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"key": "value"}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 2
    var_10 = lambda x, y: (var_8, y + var_9)
    var_11 = ''
    var_12 = 0
    var_13 = (var_11, var_12)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '[]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/14 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 1

def test_case_0():
    var_0 = '{"test": null}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": true}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": false}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": 123}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": 123.45}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": []}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": {}}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 10/12 statements.
# Partially parsed test_tokenizing_json_object_multiple_pairs. Retrieved 12/14 statements.
# Partially parsed test_tokenizing_json_object_with_whitespace. Retrieved 10/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, i: (ScalarToken(var_5 if i == var_4 else var_6, i, i + var_7, s), i + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 8
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, i: (ScalarToken(var_5 if i == var_4 else var_6, i, i + var_7, s), i + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 28/29 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 8/11 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = var_1.string
    assert var_8 == 'null'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = var_1.string
    assert var_8 == 'true'
    var_9 = 'false'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = False
    var_12 = 4
    var_13 = module_1.ScalarToken(var_11, var_11, var_12, var_9)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True
    var_15 = var_10.value
    assert var_15 is False
    var_16 = var_10.string
    assert var_16 == 'false'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = var_1.string
    assert var_8 == '42'
    var_9 = '3.14'
    var_10 = module_0.tokenize_json(var_9)
    var_11 = 3.14
    var_12 = 3
    var_13 = module_1.ScalarToken(var_11, var_3, var_12, var_9)
    var_14 = bool(var_10 == var_13)
    assert var_14 is True
    var_15 = var_10.value
    var_16 = bool(var_10.value == 3.14)
    assert var_16 is True
    var_17 = var_10.string
    assert var_17 == '3.14'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = var_1.string
    assert var_8 == '"hello"'

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = module_2.ScalarToken(var_5, var_5, var_5, var_0)
    var_18 = bool(var_16 == var_17)
    assert var_18 is True
    var_19 = [var_5]
    var_20 = var_1.lookup(var_19)
    var_21 = 2
    var_22 = 3
    var_23 = module_2.ScalarToken(var_21, var_22, var_22, var_0)
    var_24 = bool(var_20 == var_23)
    assert var_24 is True
    var_25 = [var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = 5
    var_28 = module_2.ScalarToken(var_22, var_27, var_27, var_0)
    var_29 = bool(var_26 == var_28)
    assert var_29 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = 3
    var_19 = module_2.ScalarToken(var_15, var_5, var_18, var_0)
    var_20 = bool(var_17 == var_19)
    assert var_20 is True
    var_21 = [var_15]
    var_22 = var_1.lookup(var_21)
    var_23 = 5
    var_24 = module_2.ScalarToken(var_5, var_23, var_23, var_0)
    var_25 = bool(var_22 == var_24)
    assert var_25 is True
    var_26 = 'b'
    var_27 = [var_26]
    var_28 = var_1.lookup_key(var_27)
    var_29 = 8
    var_30 = 10
    var_31 = module_2.ScalarToken(var_26, var_29, var_30, var_0)
    var_32 = bool(var_28 == var_31)
    assert var_32 is True
    var_33 = [var_26]
    var_34 = var_1.lookup(var_33)
    var_35 = 2
    var_36 = 12
    var_37 = module_2.ScalarToken(var_35, var_36, var_36, var_0)
    var_38 = bool(var_34 == var_37)
    assert var_38 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_6.value == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    var_13 = bool(var_11.value == {'c': 3})
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_tokenize_json_with_valid_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 19/21 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 13/15 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_0)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'value'
    var_2 = 7
    var_3 = 13
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 14
    var_6 = (var_4, var_5)
    var_7 = lambda s, e: var_6
    var_8 = '{"key": "value"}'
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = True
    var_12 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 8
    var_2 = 'value1'
    var_3 = 15
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)
    var_5 = 16
    var_6 = (var_4, var_5)
    var_7 = 'value2'
    var_8 = 25
    var_9 = 32
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = 33
    var_12 = (var_10, var_11)
    var_13 = lambda s, e: var_6 if e == var_1 else var_12
    var_14 = '{"key1": "value1", "key2": "value2"}'
    var_15 = 0
    var_16 = (var_14, var_15)
    var_17 = True
    var_18 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 'value'
    var_2 = 9
    var_3 = 15
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 16
    var_6 = (var_4, var_5)
    var_7 = lambda s, e: var_6
    var_8 = '{"key":  "value"}'
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = True
    var_12 = {}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 13
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 14
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_10, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 8
    var_6 = 15
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 16
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_10, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 13
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 14
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_10, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 13
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 14
    var_9 = (var_7, var_8)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_10, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_parse_object_is_not_tokenizing_json_object. Retrieved 1/5 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = ''
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_0)
    var_6 = (var_5, var_1)
    var_7 = lambda s, end: var_6
    var_8 = {}
    var_9 = ''
    var_10 = module_1._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_9)
    var_11 = bool(var_10 == ({}, 0))
    assert var_11 is True



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    assert var_4 == ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 10/11 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = lambda s, i: (ScalarToken(var_4, var_1, var_1, s), i + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 7
    var_4 = 12
    var_5 = 13
    var_6 = lambda s, i: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = True

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, i: (ScalarToken(var_2, i, i + var_3, s), i + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 9
    var_4 = 14
    var_5 = 15
    var_6 = lambda s, i: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 8
    var_4 = 13
    var_5 = 14
    var_6 = lambda s, i: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = True
    var_10 = module_0._TokenizingJSONObject(var_8, var_9, var_6, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, i: (ScalarToken(var_2, i, i + var_3, s), i + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = module_0._TokenizingJSONObject(var_7, var_8, var_5, var_1, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 'value'
    var_3 = 7
    var_4 = 12
    var_5 = 13
    var_6 = lambda s, i: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = True
    var_10 = module_0._TokenizingJSONObject(var_8, var_9, var_6, var_1, var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_IndexError_handling_in_whitespace_optimization. Retrieved 8/15 statements.


def test_case_0():
    var_0 = '{"key":value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = 1
    var_5 = var_3 + var_4
    var_6 = 1
    var_7 = var_5 + var_6
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 9
    var_6 = 'value1'
    var_7 = 6
    var_8 = 'value2'
    var_9 = 7
    var_10 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s) if e == var_5 else ScalarToken(var_8, e, e + var_7, s), e + var_9)
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_2, var_4, var_8, var_3, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 10
    var_6 = 'value1'
    var_7 = 6
    var_8 = 'value2'
    var_9 = 7
    var_10 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s) if e == var_5 else ScalarToken(var_8, e, e + var_7, s), e + var_9)
    var_11 = module_0._TokenizingJSONObject(var_2, var_4, var_10, var_3, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_2, var_4, var_8, var_3, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_index_error_in_whitespace_skipping. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = 8
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = 1
    var_5 = var_3 + var_4
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_IndexError_handling_in_whitespace_skipping. Retrieved 14/18 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = 'Match'
    var_5 = ()
    var_6 = 'end'
    var_7 = lambda s, end: type(var_4, var_5, {var_6: lambda self: end})(s, end)
    var_8 = ' '
    var_9 = 1
    var_10 = var_3 + var_9
    var_11 = 1
    var_12 = var_10 + var_11
    var_13 = var_7(var_0, var_12)
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = 'value'
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, var_0), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = True
    var_8 = len(var_0)
    var_9 = bool(var_1 == var_8)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    assert var_4 == '"'
    var_5 = 1
    var_6 = var_3 + var_5
    var_7 = var_0[var_6]
    var_8 = ''
    assert var_8 == ''



# Parsed testcases at query #36
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 9/10 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = None
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, var_0), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_parse_object_is_not_tokenizing_json_object. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object_token. Retrieved 2/21 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = '"test"'
    var_2 = 0

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

def test_case_0():
    var_0 = '[]'
    var_1 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'memo'
    var_7 = 'strict'
    var_8 = None
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = {}
    var_16 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 5/9 statements.
# Partially parsed test__TokenizingJSONObject_memoization. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Expecting value'

def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = bool('key' in var_0)
    assert var_6 is True
    var_7 = var_0['key']
    assert var_7 == 'key'



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 7
    var_2 = var_0[var_1]
    assert var_2 == ' '
    var_3 = var_0[var_1 + 1]
    assert var_3 == ' '



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 7/17 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 11/21 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 4/13 statements.
# Partially parsed test_make_scanner_scans_float_number. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = '"value"'
    var_3 = 0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'key'
    var_4 = 3
    var_5 = '{"key": "value"}'
    var_6 = len(var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = 'value'
    var_8 = 5
    var_9 = '[1]'
    var_10 = len(var_9)

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = 'value'
    var_1 = 4
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = 'value'
    var_1 = 3
    var_2 = '123'
    var_3 = 0

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = '123.45'
    var_3 = 0



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 7
    var_2 = var_0[var_1:var_1 + 1]
    var_3 = bool(var_0[var_1:var_1 + 1] != ':')
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    var_5 = bool(var_4 != '')
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 22/30 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 20/26 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 5
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'

def test_case_0():
    var_0 = {}
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y, z, w, content: var_2
    var_4 = 'MockContext'
    var_5 = ()
    var_6 = 'parse_array'
    var_7 = 'parse_string'
    var_8 = 'parse_float'
    var_9 = 'parse_int'
    var_10 = 'strict'
    var_11 = 'memo'
    var_12 = []
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = lambda x, y: var_14
    var_16 = ''
    var_17 = (var_16, var_13)
    var_18 = lambda x, y, z: var_17
    var_19 = False
    var_20 = {}
    var_21 = '{}'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 1
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = 0
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '[]'

def test_case_0():
    var_0 = 'test'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'MockContext'
    var_4 = ()
    var_5 = 'parse_array'
    var_6 = 'parse_string'
    var_7 = 'parse_float'
    var_8 = 'parse_int'
    var_9 = 'strict'
    var_10 = 'memo'
    var_11 = []
    var_12 = 0
    var_13 = (var_11, var_12)
    var_14 = lambda x, y: var_13
    var_15 = ''
    var_16 = (var_15, var_12)
    var_17 = lambda x, y, z: var_16
    var_18 = False
    var_19 = len(var_2)
    assert var_19 == 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 28/29 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 8/11 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = var_1.string
    assert var_8 == 'null'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = var_1.string
    assert var_8 == 'true'
    var_9 = module_2.Position(var_2, var_2, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 4
    var_13 = module_2.Position(var_2, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = bool(var_1 == var_4)
    assert var_5 is True
    var_6 = var_1.value
    assert var_6 is False
    var_7 = var_1.string
    assert var_7 == 'false'
    var_8 = 1
    var_9 = module_2.Position(var_8, var_8, var_2)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 5
    var_13 = module_2.Position(var_8, var_12, var_3)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = var_1.string
    assert var_8 == '42'
    var_9 = module_2.Position(var_4, var_4, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 2
    var_13 = module_2.Position(var_4, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 3.14
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    var_8 = bool(var_1.value == 3.14)
    assert var_8 is True
    var_9 = var_1.string
    assert var_9 == '3.14'
    var_10 = 1
    var_11 = module_2.Position(var_10, var_10, var_3)
    var_12 = var_1.start
    var_13 = bool(var_1.start == var_11)
    assert var_13 is True
    var_14 = 4
    var_15 = module_2.Position(var_10, var_14, var_4)
    var_16 = var_1.end
    var_17 = bool(var_1.end == var_15)
    assert var_17 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = var_1.string
    assert var_8 == '"hello"'
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 7
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 8
    var_11 = 7
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = [var_6]
    var_16 = var_1.lookup(var_15)
    var_17 = module_2.ScalarToken(var_5, var_5, var_5, var_0)
    var_18 = bool(var_16 == var_17)
    assert var_18 is True
    var_19 = [var_5]
    var_20 = var_1.lookup(var_19)
    var_21 = 2
    var_22 = 3
    var_23 = module_2.ScalarToken(var_21, var_22, var_22, var_0)
    var_24 = bool(var_20 == var_23)
    assert var_24 is True
    var_25 = [var_21]
    var_26 = var_1.lookup(var_25)
    var_27 = 5
    var_28 = module_2.ScalarToken(var_22, var_27, var_27, var_0)
    var_29 = bool(var_26 == var_28)
    assert var_29 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 15
    var_11 = 14
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True
    var_15 = 'a'
    var_16 = [var_15]
    var_17 = var_1.lookup_key(var_16)
    var_18 = 3
    var_19 = module_2.ScalarToken(var_15, var_5, var_18, var_0)
    var_20 = bool(var_17 == var_19)
    assert var_20 is True
    var_21 = [var_15]
    var_22 = var_1.lookup(var_21)
    var_23 = 5
    var_24 = module_2.ScalarToken(var_5, var_23, var_23, var_0)
    var_25 = bool(var_22 == var_24)
    assert var_25 is True
    var_26 = 'b'
    var_27 = [var_26]
    var_28 = var_1.lookup_key(var_27)
    var_29 = 8
    var_30 = 10
    var_31 = module_2.ScalarToken(var_26, var_29, var_30, var_0)
    var_32 = bool(var_28 == var_31)
    assert var_32 is True
    var_33 = [var_26]
    var_34 = var_1.lookup(var_33)
    var_35 = 2
    var_36 = 12
    var_37 = module_2.ScalarToken(var_35, var_36, var_36, var_0)
    var_38 = bool(var_34 == var_37)
    assert var_38 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"a": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': [1, 2], 'nested': {'a': 3}})
    assert var_3 is True
    var_4 = 'list'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    var_8 = bool(var_6.value == [1, 2])
    assert var_8 is True
    var_9 = 'nested'
    var_10 = [var_9]
    var_11 = var_1.lookup(var_10)
    var_12 = var_11.value
    var_13 = bool(var_11.value == {'a': 3})
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_index_error_handling. Retrieved 8/9 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = None
    var_3 = lambda s, end: (ScalarToken(var_2, end, end, s), end)
    var_4 = 0
    var_5 = (var_0, var_4)
    var_6 = True
    var_7 = {}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_whitespace_after_colon_with_two_spaces. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '{"key":  value}'
    var_1 = 6
    var_2 = var_0[var_1]
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_4]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_index_error_handling_in_tokenizing_json_object. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = 'value'
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, var_0), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = False
    var_8 = len(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/12 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = 'a'
    var_9 = module_0.ScalarToken(var_3, var_1, var_1)
    var_10 = {var_8: var_9}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a":1,"b":2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = module_0.ScalarToken(var_3, var_1, var_1)
    var_11 = 2
    var_12 = module_0.ScalarToken(var_11, var_1, var_1)
    var_13 = {var_8: var_10, var_9: var_12}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "a" : 1 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = 'a'
    var_9 = module_0.ScalarToken(var_3, var_1, var_1)
    var_10 = {var_8: var_9}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"a" 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = module_1._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"a":1 "b":2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = module_1._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"a":1'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = module_1._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{1:2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = module_0.ScalarToken(var_3, var_1, var_1)
    var_5 = (var_4, var_3)
    var_6 = lambda s, e: var_5
    var_7 = {}
    var_8 = module_1._TokenizingJSONObject(var_2, var_3, var_6, var_7, var_0)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    assert var_3 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 21/29 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 18/26 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 5
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '"test"'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = 1
    var_9 = 0
    var_10 = ''
    var_11 = module_0.ScalarToken(var_8, var_9, var_9, var_10)
    var_12 = [var_11]
    var_13 = 3
    var_14 = (var_12, var_13)
    var_15 = lambda x, y: var_14
    var_16 = (var_10, var_9)
    var_17 = lambda x, y, z: var_16
    var_18 = False
    var_19 = {}
    var_20 = '[1]'

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 5
    var_14 = lambda x, y, z: (var_12, y + var_13)
    var_15 = False
    var_16 = {}
    var_17 = '{"key":1}'



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 10
    var_2 = ' \t\n\r'
    var_3 = '\\s*'
    var_4 = lambda s, end: re.match(var_3, s[end:])
    var_5 = var_0[var_1]
    var_6 = var_5 in var_2



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skip. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = 'value'
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, s), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = True
    var_8 = len(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 16/27 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/17 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = module_0.ScalarToken(var_4, var_1, var_1, var_0)
    var_6 = (var_5, var_3)
    var_7 = lambda s, e: var_6
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = 'key'
    var_7 = 4
    var_8 = module_0.ScalarToken(var_6, var_4, var_7, var_0)
    var_9 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = [var_1]
    var_3 = '{"key1": "value1", "key2": "value2"}'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = True
    var_7 = {}
    var_8 = 'key1'
    var_9 = 6
    var_10 = module_0.ScalarToken(var_8, var_6, var_9, var_0)
    var_11 = 'key2'
    var_12 = 16
    var_13 = 21
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = '{"key" : "value"}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = 'key'
    var_7 = 4
    var_8 = module_0.ScalarToken(var_6, var_4, var_7, var_0)
    var_9 = len(var_0)

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 0
    var_6 = ''
    var_7 = module_0.ScalarToken(var_4, var_5, var_5, var_6)
    var_8 = (var_7, var_3)
    var_9 = lambda s, e: var_8
    var_10 = {}
    var_11 = module_1._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = '{"key1": "value1" "key2": "value2"}'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = None
    var_6 = 0
    var_7 = ''
    var_8 = module_0.ScalarToken(var_5, var_6, var_6, var_7)
    var_9 = (var_8, var_4)
    var_10 = lambda s, e: var_9
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_3, var_4, var_10, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = "{key: 'value'}"
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 0
    var_6 = ''
    var_7 = module_0.ScalarToken(var_4, var_5, var_5, var_6)
    var_8 = (var_7, var_3)
    var_9 = lambda s, e: var_8
    var_10 = {}
    var_11 = module_1._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 24/30 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 24/30 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 23/29 statements.
# Partially parsed test_make_scanner_raises_stop_iteration. Retrieved 19/25 statements.


def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'test'
    var_13 = 6
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '"test"'
    var_19 = 1
    var_20 = module_0.Position(var_19, var_19, var_16)
    var_21 = 5
    var_22 = module_0.Position(var_19, var_13, var_21)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = 'key'
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": "value"}'
    var_19 = 1
    var_20 = module_0.Position(var_19, var_19, var_16)
    var_21 = 15
    var_22 = 14
    var_23 = module_0.Position(var_19, var_21, var_22)

import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = 1
    var_9 = 0
    var_10 = ''
    var_11 = module_0.ScalarToken(var_8, var_9, var_9, var_10)
    var_12 = [var_11]
    var_13 = 3
    var_14 = (var_12, var_13)
    var_15 = lambda x, y: var_14
    var_16 = (var_10, var_9)
    var_17 = lambda x, y, z: var_16
    var_18 = False
    var_19 = {}
    var_20 = '[1]'
    var_21 = module_1.Position(var_8, var_8, var_18)
    var_22 = 2
    var_23 = module_1.Position(var_8, var_13, var_22)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'null'
    var_18 = 1
    var_19 = module_0.Position(var_18, var_18, var_15)
    var_20 = 4
    var_21 = 3
    var_22 = module_0.Position(var_18, var_20, var_21)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'true'
    var_18 = 1
    var_19 = module_0.Position(var_18, var_18, var_15)
    var_20 = 4
    var_21 = 3
    var_22 = module_0.Position(var_18, var_20, var_21)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = 'false'
    var_18 = 1
    var_19 = module_0.Position(var_18, var_18, var_15)
    var_20 = 5
    var_21 = 4
    var_22 = module_0.Position(var_18, var_20, var_21)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123'
    var_18 = 1
    var_19 = module_0.Position(var_18, var_18, var_15)
    var_20 = 3
    var_21 = 2
    var_22 = module_0.Position(var_18, var_20, var_21)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = '123.45'
    var_18 = 1
    var_19 = module_0.Position(var_18, var_18, var_15)
    var_20 = 6
    var_21 = 5
    var_22 = module_0.Position(var_18, var_20, var_21)

def test_case_0():
    var_0 = 'MockContext'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = []
    var_9 = 0
    var_10 = (var_8, var_9)
    var_11 = lambda x, y: var_10
    var_12 = ''
    var_13 = (var_12, var_9)
    var_14 = lambda x, y, z: var_13
    var_15 = False
    var_16 = {}
    var_17 = ''
    var_18 = 0
    var_19 = bool(False)
    assert var_19 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 24/25 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 16/17 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 8
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key": "value"}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = (var_8, var_7)
    var_14 = lambda s, e: var_13
    var_15 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 'key1'
    var_2 = 1
    var_3 = 5
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value1'
    var_6 = 9
    var_7 = 15
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'key2'
    var_10 = 18
    var_11 = 22
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 26
    var_15 = 32
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = '{"key1": "value1", "key2": "value2"}'
    var_18 = 0
    var_19 = (var_17, var_18)
    var_20 = True
    var_21 = 8
    var_22 = lambda s, e: (var_8 if e == var_21 else var_16, var_15)
    var_23 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 10
    var_7 = 15
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key":  "value"}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = (var_8, var_7)
    var_14 = lambda s, e: var_13
    var_15 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 9/12 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 13/17 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 13/17 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 12/16 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 13/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = 'test'
    var_5 = 6
    var_6 = (var_4, var_5)
    var_7 = lambda x, y, z: var_6
    var_8 = {}
    var_9 = True
    var_10 = '{"key": "test"}'
    var_11 = '"test"'
    var_12 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True
    var_9 = 'null'
    var_10 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True
    var_9 = 'true'
    var_10 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True
    var_9 = 'false'
    var_10 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True
    var_9 = '123'
    var_10 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = {}
    var_8 = True
    var_9 = '123.45'
    var_10 = 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = ''
    var_3 = module_0.ScalarToken(var_0, var_1, var_1, var_2)
    var_4 = [var_3]
    var_5 = 3
    var_6 = (var_4, var_5)
    var_7 = lambda x, y: var_6
    var_8 = (var_2, var_0)
    var_9 = lambda x, y, z: var_8
    var_10 = {}
    var_11 = True
    var_12 = '[1]'

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = 'key'
    var_5 = 5
    var_6 = (var_4, var_5)
    var_7 = lambda x, y, z: var_6
    var_8 = {}
    var_9 = True
    var_10 = '{"key": 1}'
    var_11 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = ''
    var_5 = (var_4, var_1)
    var_6 = lambda x, y, z: var_5
    var_7 = 'test'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = True
    var_11 = 'null'
    var_12 = 0



# Parsed testcases at query #26
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = 'invalid json'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = '{"key": value}'
    var_1 = 10
    var_2 = ''
    var_3 = ()
    var_4 = 'end'
    var_5 = lambda s, end: type(var_2, var_3, {var_4: lambda : end})()
    var_6 = ' '
    var_7 = bool(not var_0[var_1] in var_6)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 26/27 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 21/22 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 9/10 statements.
# Partially parsed test_tokenize_json_invalid_json. Retrieved 2/5 statements.


import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = None
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is None
    var_8 = 1
    var_9 = module_2.Position(var_8, var_8, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 4
    var_13 = module_2.Position(var_8, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = True
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 is True
    var_8 = module_2.Position(var_2, var_2, var_3)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 4
    var_12 = module_2.Position(var_2, var_11, var_4)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = False
    var_3 = 4
    var_4 = module_1.ScalarToken(var_2, var_2, var_3, var_0)
    var_5 = bool(var_1 == var_4)
    assert var_5 is True
    var_6 = var_1.value
    assert var_6 is False
    var_7 = 1
    var_8 = module_2.Position(var_7, var_7, var_2)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 5
    var_12 = module_2.Position(var_7, var_11, var_3)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 42
    var_3 = 0
    var_4 = 1
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 42
    var_8 = module_2.Position(var_4, var_4, var_3)
    var_9 = var_1.start
    var_10 = bool(var_1.start == var_8)
    assert var_10 is True
    var_11 = 2
    var_12 = module_2.Position(var_4, var_11, var_4)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 3.14
    var_3 = 0
    var_4 = 3
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    var_8 = bool(var_1.value == 3.14)
    assert var_8 is True
    var_9 = 1
    var_10 = module_2.Position(var_9, var_9, var_3)
    var_11 = var_1.start
    var_12 = bool(var_1.start == var_10)
    assert var_12 is True
    var_13 = 4
    var_14 = module_2.Position(var_9, var_13, var_4)
    var_15 = var_1.end
    var_16 = bool(var_1.end == var_14)
    assert var_16 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'hello'
    var_3 = 0
    var_4 = 6
    var_5 = module_1.ScalarToken(var_2, var_3, var_4, var_0)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True
    var_7 = var_1.value
    assert var_7 == 'hello'
    var_8 = 1
    var_9 = module_2.Position(var_8, var_8, var_3)
    var_10 = var_1.start
    var_11 = bool(var_1.start == var_9)
    assert var_11 is True
    var_12 = 7
    var_13 = module_2.Position(var_8, var_12, var_4)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 8
    var_10 = 7
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True
    var_14 = [var_5]
    var_15 = var_1.lookup(var_14)
    var_16 = module_2.ScalarToken(var_4, var_4, var_4, var_0)
    var_17 = bool(var_15 == var_16)
    assert var_17 is True
    var_18 = [var_4]
    var_19 = var_1.lookup(var_18)
    var_20 = 2
    var_21 = 3
    var_22 = module_2.ScalarToken(var_20, var_21, var_21, var_0)
    var_23 = bool(var_19 == var_22)
    assert var_23 is True
    var_24 = [var_20]
    var_25 = var_1.lookup(var_24)
    var_26 = 5
    var_27 = module_2.ScalarToken(var_21, var_26, var_26, var_0)
    var_28 = bool(var_25 == var_27)
    assert var_28 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1
import typesystem.tokenize.tokens as module_2

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 15
    var_10 = 14
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True
    var_14 = 'a'
    var_15 = [var_14]
    var_16 = var_1.lookup_key(var_15)
    var_17 = module_2.ScalarToken(var_14, var_4, var_4, var_0)
    var_18 = bool(var_16 == var_17)
    assert var_18 is True
    var_19 = [var_14]
    var_20 = var_1.lookup(var_19)
    var_21 = 5
    var_22 = module_2.ScalarToken(var_4, var_21, var_21, var_0)
    var_23 = bool(var_20 == var_22)
    assert var_23 is True
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = var_1.lookup_key(var_25)
    var_27 = 8
    var_28 = module_2.ScalarToken(var_24, var_27, var_27, var_0)
    var_29 = bool(var_26 == var_28)
    assert var_29 is True
    var_30 = [var_24]
    var_31 = var_1.lookup(var_30)
    var_32 = 2
    var_33 = 12
    var_34 = module_2.ScalarToken(var_32, var_33, var_33, var_0)
    var_35 = bool(var_31 == var_34)
    assert var_35 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": 3}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    var_8 = 1
    var_9 = 7
    var_10 = module_1.ScalarToken(var_8, var_9, var_9, var_0)
    var_11 = bool(var_7 == var_10)
    assert var_11 is True
    var_12 = [var_4, var_8]
    var_13 = var_1.lookup(var_12)
    var_14 = 2
    var_15 = 9
    var_16 = module_1.ScalarToken(var_14, var_15, var_15, var_0)
    var_17 = bool(var_13 == var_16)
    assert var_17 is True
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_18, var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = 3
    var_23 = 20
    var_24 = module_1.ScalarToken(var_22, var_23, var_23, var_0)
    var_25 = bool(var_21 == var_24)
    assert var_25 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.tokenize.tokens as module_1

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = 1
    var_8 = 6
    var_9 = '{"a": 1}'
    var_10 = module_1.ScalarToken(var_7, var_8, var_8, var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a":}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 18/20 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 12/14 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = ''
    var_6 = module_0.ScalarToken(var_4, var_1, var_1, var_5)
    var_7 = lambda s, end: (var_6, end)
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'value'
    var_2 = 7
    var_3 = 13
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 14
    var_6 = (var_4, var_5)
    var_7 = lambda s, end: var_6
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 8
    var_2 = 'value1'
    var_3 = 14
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = 'value2'
    var_8 = 24
    var_9 = 30
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = 31
    var_12 = (var_10, var_11)
    var_13 = lambda s, end: var_6 if end == var_1 else var_12
    var_14 = 0
    var_15 = (var_0, var_14)
    var_16 = True
    var_17 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = ' { "key" : "value" } '
    var_1 = 'value'
    var_2 = 12
    var_3 = 18
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 19
    var_6 = (var_4, var_5)
    var_7 = lambda s, end: var_6
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = {}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 'value'
    var_2 = 8
    var_3 = 14
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = lambda s, end: var_6
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_9, var_10, var_7, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 9
    var_2 = 'value1'
    var_3 = 15
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)
    var_5 = 16
    var_6 = (var_4, var_5)
    var_7 = 'value2'
    var_8 = 25
    var_9 = 31
    var_10 = module_0.ScalarToken(var_7, var_8, var_9, var_0)
    var_11 = 32
    var_12 = (var_10, var_11)
    var_13 = lambda s, end: var_6 if end == var_1 else var_12
    var_14 = 0
    var_15 = (var_0, var_14)
    var_16 = True
    var_17 = {}
    var_18 = module_1._TokenizingJSONObject(var_15, var_16, var_13, var_17, var_0)
    var_19 = bool(False)
    assert var_19 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 'value'
    var_2 = 8
    var_3 = 14
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 15
    var_6 = (var_4, var_5)
    var_7 = lambda s, end: var_6
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_9, var_10, var_7, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 'value'
    var_2 = 7
    var_3 = 13
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 14
    var_6 = (var_4, var_5)
    var_7 = lambda s, end: var_6
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = {}
    var_12 = module_1._TokenizingJSONObject(var_9, var_10, var_7, var_11, var_0)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_tokenize_json_with_valid_json_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    var_4 = ''
    assert var_4 == ''



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 12/13 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = {}
    var_5 = 'value'
    var_6 = lambda s, end: (ScalarToken(var_5, end, end, s), end + var_2)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = len(var_0)



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = 'Expected IndexError when accessing s[end]'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_index_error_handling_in_tokenizing_json_object. Retrieved 9/10 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = {}
    var_3 = 'value'
    var_4 = lambda s, end: (ScalarToken(var_3, end, end, var_0), end)
    var_5 = 0
    var_6 = (var_0, var_5)
    var_7 = True



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    var_4 = ''
    assert var_4 == ''



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_tokenize_json_with_valid_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #40
#--------------------------




def test_case_0():
    var_0 = '{"key":'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2



