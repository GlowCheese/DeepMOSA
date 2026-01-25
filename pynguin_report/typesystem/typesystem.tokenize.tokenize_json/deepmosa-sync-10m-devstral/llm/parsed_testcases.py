####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/16 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 10/11 statements.
# Partially parsed test__TokenizingJSONObject_memoization. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = lambda s, end: (ScalarToken(var_4, var_1, var_1, s), end + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 'value'
    var_2 = 6
    var_3 = 11
    var_4 = 12
    var_5 = lambda s, end: (ScalarToken(var_1, var_2, var_3, s), var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = {}

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 7
    var_2 = 'value1'
    var_3 = 13
    var_4 = 14
    var_5 = 'value2'
    var_6 = 22
    var_7 = 28
    var_8 = 29
    var_9 = lambda s, end: (ScalarToken(var_2, var_1, var_3, s), var_4) if end == var_1 else (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = 0
    var_11 = (var_0, var_10)
    var_12 = True
    var_13 = {}

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 'value'
    var_2 = 9
    var_3 = 14
    var_4 = 15
    var_5 = lambda s, end: (ScalarToken(var_1, var_2, var_3, s), var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value",}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 12
    var_7 = 13
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 12
    var_7 = 13
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 8
    var_7 = lambda s, end: (ScalarToken(var_4, var_5, var_5, s), var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 7
    var_6 = 12
    var_7 = 13
    var_8 = lambda s, end: (ScalarToken(var_4, var_5, var_6, s), var_7)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 7
    var_4 = 12
    var_5 = 13
    var_6 = lambda s, end: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = True
    var_10 = 'key'
    var_11 = bool('key' in var_1)
    assert var_11 is True
    var_12 = var_1['key']
    assert var_12 == 'key'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/13 statements.


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
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 8
    var_3 = 'value1'
    var_4 = 6
    var_5 = 7
    var_6 = 'value2'
    var_7 = lambda s, e: (ScalarToken(var_3, e, e + var_4, s), e + var_5) if e == var_2 else (ScalarToken(var_6, e, e + var_4, s), e + var_5)
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = True
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = True
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_boolean. Retrieved 4/20 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 4/20 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 3/21 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = '"test"'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 'true'
    var_2 = 0
    var_3 = 'false'

def test_case_0():
    var_0 = ''
    var_1 = '123'
    var_2 = 0
    var_3 = '12.3'

def test_case_0():
    var_0 = ''
    var_1 = '[1, 2, 3]'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = '{"key": "value"}'
    var_2 = 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = '{"key": "value", }'
    var_1 = 13
    var_2 = var_0[var_1:var_1 + 1]
    var_3 = bool(var_0[var_1:var_1 + 1] != '"')
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = '{"key": "value", }'
    var_1 = 13
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    var_5 = bool(var_4 != '"')
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_index_error_in_whitespace_skipping. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = 7
    var_2 = var_0[var_1]
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = 1
    var_6 = var_4 + var_5
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_parse_object_is_not_tokenizing_json_object. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 8/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 8/16 statements.


def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = ''

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = '{"test": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = 'false'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: float(x)
    var_4 = lambda self, x: int(x)
    var_5 = {}
    var_6 = '123'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: float(x)
    var_4 = lambda self, x: int(x)
    var_5 = {}
    var_6 = '123.45'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: ([], 2)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = '[]'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x: (None, 0)
    var_1 = lambda self, x, y, z: (None, 0)
    var_2 = False
    var_3 = lambda self, x: 0.0
    var_4 = lambda self, x: 0
    var_5 = {}
    var_6 = '{}'
    var_7 = 0



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    assert var_4 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 8/17 statements.
# Partially parsed test_make_scanner_handles_string_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_dict_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_list_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_null_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_true_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_false_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_number_token. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_handles_float_token. Retrieved 8/15 statements.


def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = ''
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = '{"test": "value"}'
    var_7 = 1

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('key', 5)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = '{"key": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 2)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = '[]'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = True
    var_5 = {}
    var_6 = 'false'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: float(x)
    var_3 = lambda self, x: int(x)
    var_4 = True
    var_5 = {}
    var_6 = '123'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: float(x)
    var_3 = lambda self, x: int(x)
    var_4 = True
    var_5 = {}
    var_6 = '123.45'
    var_7 = 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/24 statements.


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
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/16 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"key": "test"}'
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



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    assert var_3 == ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_integer_token. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test content'

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '[1, 2, 3]'
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
    var_0 = '123.45'
    var_1 = 0

def test_case_0():
    var_0 = '123'
    var_1 = 0



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = ' \t\n\r'
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_1:var_4]
    var_6 = bool(var_5 in var_2)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = '{"key": "value", }'
    var_1 = 13
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    var_5 = bool(var_4 != '"')
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_json_simple_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_number. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_boolean. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 11/12 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

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
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": }'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 19/20 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, end: (ScalarToken(var_4, end, end, s), end + var_3)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = 'key'
    var_10 = 8
    var_11 = 12
    var_12 = module_0.ScalarToken(var_5, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = 'key1'
    var_10 = 8
    var_11 = 13
    var_12 = module_0.ScalarToken(var_5, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = 'key'
    var_10 = 8
    var_11 = 12
    var_12 = module_0.ScalarToken(var_5, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 4
    var_7 = 5
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s), end + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'inner'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 16
    var_9 = 17
    var_10 = lambda s, end: (ScalarToken(var_7, end, end + var_8, s), end + var_9)
    var_11 = 'outer'
    var_12 = {var_5: var_6}
    var_13 = 8
    var_14 = 23
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = {var_11: var_15}
    var_17 = len(var_0)



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = '  }'
    var_1 = 0
    var_2 = ' \t\n\r'
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_1:var_4]
    var_6 = bool(var_5 in var_2)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 8/9 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_1)
    var_6 = 4
    var_7 = module_0.Position(var_4, var_6, var_2)



# Parsed testcases at query #25
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key":  value}'
    var_1 = {}
    var_2 = 0
    var_3 = 'key'
    var_4 = 2
    var_5 = var_2 + var_4
    var_6 = module_0.ScalarToken(var_3, var_2, var_5, var_0)
    var_7 = var_6.string
    assert var_7 == '"key"'
    var_8 = var_6.value
    assert var_8 == 'key'
    var_9 = var_6.start.line_no
    assert var_9 == 1
    var_10 = var_6.start.column_no
    assert var_10 == 2
    var_11 = var_6.end.line_no
    assert var_11 == 1
    var_12 = var_6.end.column_no
    assert var_12 == 4



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = '  '
    var_1 = 0
    var_2 = ' \t\n\r'
    var_3 = var_0[var_1]
    var_4 = bool(var_0[var_1] in var_2)
    assert var_4 is True
    var_5 = 1
    var_6 = var_1 + var_5
    var_7 = var_0[var_6]
    var_8 = bool(var_0[var_6] in var_2)
    assert var_8 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_values. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = 1
    var_6 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_5)
    var_7 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 7
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = False
    var_5 = 8
    var_6 = 'value1'
    var_7 = 6
    var_8 = 'value2'
    var_9 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s) if e == var_5 else ScalarToken(var_8, e, e + var_7, s), e + var_5)
    var_10 = len(var_0)

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 7
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 7
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
    var_4 = False
    var_5 = 10
    var_6 = 'value1'
    var_7 = 6
    var_8 = 'value2'
    var_9 = 8
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
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 7
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_2, var_4, var_8, var_3, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 20/25 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 22/28 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 24/30 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 23/28 statements.


def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = '{"key": "test"}'
    var_21 = '"test"'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = 'null'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = 'true'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = 'false'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = '123'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = '123.45'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = 'Context'
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
    var_19 = 'clear'
    var_20 = None
    var_21 = lambda : var_20
    var_22 = {var_19: var_21}
    var_23 = '[1]'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = None
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}
    var_20 = '{"key": 1}'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'clear'
    var_17 = 'cleared'
    var_18 = None
    var_19 = lambda : var_18
    var_20 = False
    var_21 = {var_16: var_19, var_17: var_20}
    var_22 = 'null'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/10 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 10/19 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 8/17 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 2/10 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 2/10 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 2/10 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 4/12 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 4/12 statements.
# Partially parsed test_make_scanner_raises_stop_iteration. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test content'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = '{"test": "value"}'
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Position(var_4, var_5, var_4)
    var_7 = 6
    var_8 = 5
    var_9 = module_0.Position(var_4, var_7, var_8)

def test_case_0():
    var_0 = 'test'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = '{"test": "value"}'
    var_4 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = '[1, 2, 3]'
    var_7 = 0

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
    var_0 = 3.14
    var_1 = 42
    var_2 = '42'
    var_3 = 0

def test_case_0():
    var_0 = 3.14
    var_1 = 42
    var_2 = '3.14'
    var_3 = 0

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = '{"key":  "value"}'
    var_1 = 8
    var_2 = var_0[var_1]
    assert var_2 == ' '
    var_3 = var_0[var_1 + 1]
    assert var_3 == ' '
    var_4 = var_0[var_1 + 2]
    assert var_4 == '"'



# Parsed testcases at query #31
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0._make_scanner(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 5/14 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = None
    var_3 = 10
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_whitespace_handling_after_colon. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{"key":  value}'
    var_1 = 6
    var_2 = var_0[var_1]
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_4]
    var_6 = var_4 + var_3



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 11/14 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 12/17 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/14 statements.


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
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = len(var_0)

def test_case_0():
    var_0 = '  {  "key"  :  "value"  }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 6/9 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = 1
    var_3 = module_0.Position(var_2, var_1, var_1)
    var_4 = 10
    var_5 = module_0.Position(var_2, var_4, var_4)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 8/18 statements.


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



# Parsed testcases at query #37
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 11
    var_3 = 1
    var_4 = 10
    var_5 = module_1.Position(var_3, var_2, var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'null'
    var_1 = 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


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
    var_4 = 7
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s) if end == var_4 else ScalarToken(var_7, end, end + var_6, s), end + var_4)
    var_9 = {}
    var_10 = len(var_0)

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

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
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
    var_4 = 7
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = lambda s, end: (ScalarToken(var_5, end, end + var_6, s) if end == var_4 else ScalarToken(var_7, end, end + var_6, s), end + var_4)
    var_9 = {}
    var_10 = module_0._TokenizingJSONObject(var_2, var_3, var_8, var_9, var_0)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_dict. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_list. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 8/15 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 8/15 statements.


def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('', 0)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = ''

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = '{"test": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = '{"test": "value"}'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 2)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = '["test"]'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 0.0
    var_3 = lambda self, x: 0
    var_4 = False
    var_5 = {}
    var_6 = 'false'
    var_7 = 0

def test_case_0():
    var_0 = lambda self, x, y: ([], 0)
    var_1 = lambda self, x, y, z: ('test', 6)
    var_2 = lambda self, x: 123.45
    var_3 = lambda self, x: 123
    var_4 = False
    var_5 = {}
    var_6 = '123.45'
    var_7 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 10/15 statements.


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
    var_0 = '{"a": 1}'
    var_1 = 0
    assert var_1 == 7
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_4)
    var_6 = 'a'
    var_7 = 5
    var_8 = module_0.ScalarToken(var_4, var_7, var_7, var_0)
    var_9 = {var_6: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 5
    var_5 = 2
    var_6 = lambda s, e: (ScalarToken(var_3 if e == var_4 else var_5, e, e, s), e + var_3)
    var_7 = {}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = module_0.ScalarToken(var_3, var_4, var_4, var_0)
    var_11 = 12
    var_12 = module_0.ScalarToken(var_5, var_11, var_11, var_0)
    var_13 = {var_8: var_10, var_9: var_12}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a" : 1 , "b" : 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 6
    var_5 = 2
    var_6 = lambda s, e: (ScalarToken(var_3 if e == var_4 else var_5, e, e, s), e + var_3)
    var_7 = {}
    var_8 = 'a'
    var_9 = 'b'
    var_10 = module_0.ScalarToken(var_3, var_4, var_4, var_0)
    var_11 = 14
    var_12 = module_0.ScalarToken(var_5, var_11, var_11, var_0)
    var_13 = {var_8: var_10, var_9: var_12}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": {"b": 1}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = 'a'
    var_6 = {}
    var_7 = 5
    var_8 = module_0.ScalarToken(var_6, var_7, var_7, var_0)
    var_9 = {var_5: var_8}



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 6
    var_2 = var_0[var_1:var_1 + 1]
    assert var_2 == ':'



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = '{"key": "value", }'
    var_1 = len(var_0)
    var_2 = 2
    var_3 = var_1 - var_2
    var_4 = 1
    var_5 = var_3 + var_4
    var_6 = var_0[var_3:var_5]
    var_7 = bool(var_6 != '"')
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 2/20 statements.


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
    var_0 = '[1]'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 8/13 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 6/12 statements.
# Partially parsed test_make_scanner_scans_array_token. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test content'

import typesystem.base as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = '"test"'
    var_3 = 0
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_3)
    var_6 = 4
    var_7 = module_0.Position(var_4, var_1, var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 4
    var_5 = 3
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 4
    var_5 = 3
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 5
    var_5 = 4
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = '42'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 2
    var_5 = module_0.Position(var_2, var_4, var_2)

import typesystem.base as module_0

def test_case_0():
    var_0 = []
    var_1 = 2
    var_2 = '[]'
    var_3 = 0
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_3)
    var_6 = module_0.Position(var_4, var_4, var_4)

import typesystem.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = 2
    var_2 = '{}'
    var_3 = 0
    var_4 = 1
    var_5 = module_0.Position(var_4, var_4, var_3)
    var_6 = module_0.Position(var_4, var_4, var_4)

def test_case_0():
    var_0 = 'null'
    var_1 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/24 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/27 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/26 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 21/29 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 19/27 statements.


def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_18 = '"test"'

def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key":1}'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 11/12 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

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
    var_8 = 'false'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = False
    var_11 = 4
    var_12 = module_1.ScalarToken(var_10, var_10, var_11, var_8)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True
    var_14 = var_9.value
    assert var_14 is False

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
    var_8 = '3.14'
    var_9 = module_0.tokenize_json(var_8)
    var_10 = 3.14
    var_11 = 3
    var_12 = module_1.ScalarToken(var_10, var_3, var_11, var_8)
    var_13 = bool(var_9 == var_12)
    assert var_13 is True
    var_14 = var_9.value
    var_15 = bool(var_9.value == 3.14)
    assert var_15 is True

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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

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
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 23/24 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 16/17 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
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
    var_9 = 0
    var_10 = (var_0, var_9)
    var_11 = True
    var_12 = 14
    var_13 = (var_8, var_12)
    var_14 = lambda s, i: var_13
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
    var_10 = 19
    var_11 = 23
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'value2'
    var_14 = 27
    var_15 = 33
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = 0
    var_18 = (var_0, var_17)
    var_19 = True
    var_20 = 34
    var_21 = lambda s, i: (var_8 if i == var_6 else var_16, var_20)
    var_22 = {}

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
    var_9 = 0
    var_10 = (var_0, var_9)
    var_11 = True
    var_12 = 16
    var_13 = (var_8, var_12)
    var_14 = lambda s, i: var_13
    var_15 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, var_0), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key":}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, var_0), i + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 12/13 statements.


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = {}
    var_5 = 'value'
    var_6 = lambda s, end: (ScalarToken(var_5, end, end, var_0), end + var_2)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = len(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = None
    var_1 = ''
    var_2 = module_0._make_scanner(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #15
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
    var_0 = '[1]'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": 1}'
    var_1 = 0



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True
    var_14 = var_4.string
    assert var_14 == 'null'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = ''
    assert var_0 == ''
    var_1 = 1



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = '{"key":  value}'
    var_1 = 6
    var_2 = var_0[var_1]
    assert var_2 == ' '
    var_3 = 1
    var_4 = var_1 + var_3
    assert var_4 == 7
    var_5 = var_0[var_4]
    assert var_5 == ' '



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 2/20 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/15 statements.


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
    var_0 = '{"test": 42}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": 42.5}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": []}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": {}}'
    var_1 = 9

def test_case_0():
    var_0 = '{"test": "value"}'
    var_1 = 1



# Parsed testcases at query #20
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
    var_11 = ''
    assert var_11 == ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_list. Retrieved 18/19 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 19/20 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

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
    var_10 = 17
    var_11 = 16
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
    var_16 = var_1.lookup_key(var_15)
    var_17 = var_16.value
    assert var_17 == 'c'
    var_18 = [var_13, var_14]
    var_19 = var_1.lookup(var_18)
    var_20 = var_19.value
    assert var_20 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"a": 1}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1}'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1,}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_empty_string. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_quoted_string. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_object. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_array. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_null. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_number. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 3/7 statements.
# Partially parsed test_make_scanner_with_invalid_input. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test content'

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '"value"'
    var_2 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '{"key": "value"}'
    var_2 = 0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = '[1, 2, 3]'
    var_2 = 0

def test_case_0():
    var_0 = 'null'
    var_1 = 'null'
    var_2 = 0

def test_case_0():
    var_0 = 'true'
    var_1 = 'true'
    var_2 = 0

def test_case_0():
    var_0 = 'false'
    var_1 = 'false'
    var_2 = 0

def test_case_0():
    var_0 = '123'
    var_1 = '123'
    var_2 = 0

def test_case_0():
    var_0 = '123.45'
    var_1 = '123.45'
    var_2 = 0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'invalid'
    var_2 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_boolean. Retrieved 4/6 statements.
# Partially parsed test_tokenize_json_number. Retrieved 4/6 statements.
# Partially parsed test_tokenize_json_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_list. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 13/14 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.
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

def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 'false'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    assert var_6 is False
    var_7 = var_5.string
    assert var_7 == 'false'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = '3.14'
    var_5 = module_0.tokenize_json(var_4)
    var_6 = var_5.value
    var_7 = bool(var_5.value == 3.14)
    assert var_7 is True
    var_8 = var_5.string
    assert var_8 == '3.14'

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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = var_1._value
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 0
    var_8 = [var_7]
    var_9 = var_1.lookup(var_8)
    var_10 = var_9.value
    assert var_10 == 1
    var_11 = 1
    var_12 = [var_11]
    var_13 = var_1.lookup(var_12)
    var_14 = var_13.value
    assert var_14 == 2
    var_15 = 2
    var_16 = [var_15]
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 3

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': 2})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": 1, "b": 2}'
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = var_1.lookup_key(var_6)
    var_8 = var_7.value
    assert var_8 == 'a'
    var_9 = [var_5]
    var_10 = var_1.lookup(var_9)
    var_11 = var_10.value
    assert var_11 == 1
    var_12 = 'b'
    var_13 = [var_12]
    var_14 = var_1.lookup(var_13)
    var_15 = var_14.value
    assert var_15 == 2

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
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 8/15 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = 1
    var_5 = var_3 + var_4
    var_6 = 1
    var_7 = var_5 + var_6
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 8/12 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = var_0[var_3]
    var_5 = 1
    var_6 = var_3 + var_5
    var_7 = var_0[var_3]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(var_7 != '')
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 12/13 statements.


def test_case_0():
    var_0 = '{"key": "value"'
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



# Parsed testcases at query #28
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    assert var_5 == 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 13/17 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 10
    var_2 = 'Match'
    var_3 = ()
    var_4 = 'end'
    var_5 = 1
    var_6 = lambda s, end: type(var_2, var_3, {var_4: lambda self: end + var_5})()
    var_7 = ' \t\n\r'
    var_8 = 1
    var_9 = var_1 + var_8
    var_10 = 1
    var_11 = var_9 + var_10
    var_12 = var_6(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/24 statements.


def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_18 = '"test"'

def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": 1}'

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 6/7 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 14/15 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, var_1, var_1, s), e)
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = '{"a": 1}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = lambda s, e: (ScalarToken(var_4, var_2, var_2, s), e)
    var_6 = {}
    var_7 = 'a'
    var_8 = module_0.ScalarToken(var_4, var_2, var_2, var_0)
    var_9 = {var_7: var_8}
    var_10 = len(var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = '{"a": 1, "b": 2}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = lambda s, e: (ScalarToken(var_4, var_2, var_2, s), e)
    var_6 = {}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = module_0.ScalarToken(var_4, var_2, var_2, var_0)
    var_10 = 2
    var_11 = module_0.ScalarToken(var_10, var_2, var_2, var_0)
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = len(var_1)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a" : 1 , "b" : 2}'
    var_1 = '{"a" : 1 , "b" : 2}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = lambda s, e: (ScalarToken(var_4, var_2, var_2, s), e)
    var_6 = {}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = module_0.ScalarToken(var_4, var_2, var_2, var_0)
    var_10 = 2
    var_11 = module_0.ScalarToken(var_10, var_2, var_2, var_0)
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = len(var_1)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a" 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, var_1, var_1, s), e)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, var_1, var_1, s), e)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, var_1, var_1, s), e)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{1: 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, var_1, var_1, s), e)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = ''
    var_3 = var_0[var_1]
    var_4 = ''
    assert var_4 == ''



# Parsed testcases at query #35
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = None
    var_2 = 8
    var_3 = 11
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = var_4.value
    assert var_5 is None
    var_6 = var_4.start._index
    assert var_6 == 8
    var_7 = var_4.end._index
    assert var_7 == 11
    var_8 = var_4.string
    assert var_8 == 'null'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 12/13 statements.


def test_case_0():
    var_0 = '{"key": "value"'
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 21/27 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 20/25 statements.


def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_18 = '{"key": "test"}'
    var_19 = '"test"'

def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": "value"}'
    var_19 = len(var_18)

def test_case_0():
    var_0 = 'Context'
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
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = 'null'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 19/25 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_array'
    var_4 = 'parse_string'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
    var_7 = 'memo'
    var_8 = 'strict'
    var_9 = []
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = lambda x, y: var_11
    var_13 = ''
    var_14 = (var_13, var_10)
    var_15 = lambda x, y, z: var_14
    var_16 = {}
    var_17 = True
    var_18 = 8



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_tokenizing_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenizing_single_key_value_pair. Retrieved 11/13 statements.
# Partially parsed test_tokenizing_multiple_key_value_pairs. Retrieved 7/13 statements.
# Partially parsed test_tokenizing_with_whitespace. Retrieved 12/14 statements.


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
    var_1 = '{"key": "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = {}
    var_10 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = '{"key1": "value1", "key2": "value2"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = {}
    var_6 = len(var_0)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = '  {"key": "value"}  '
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = {}
    var_10 = len(var_0)
    var_11 = var_10 + var_2

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
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
    var_4 = 'value1'
    var_5 = 7
    var_6 = 8
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = 10
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 19/20 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 15/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = 1
    var_6 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_5)
    var_7 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 1
    var_7 = (var_0, var_6)
    var_8 = False
    var_9 = 'key'
    var_10 = 7
    var_11 = 11
    var_12 = module_0.ScalarToken(var_2, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 7
    var_3 = 'value1'
    var_4 = 6
    var_5 = 'value2'
    var_6 = lambda s, e: (ScalarToken(var_3, e, e + var_4, s), e + var_2) if e == var_2 else (ScalarToken(var_5, e, e + var_4, s), e + var_2)
    var_7 = 1
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = 'key1'
    var_11 = 'key2'
    var_12 = 12
    var_13 = module_0.ScalarToken(var_3, var_2, var_12, var_0)
    var_14 = 21
    var_15 = 26
    var_16 = module_0.ScalarToken(var_5, var_14, var_15, var_0)
    var_17 = {var_10: var_13, var_11: var_16}
    var_18 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 1
    var_7 = (var_0, var_6)
    var_8 = False
    var_9 = 'key'
    var_10 = 8
    var_11 = 12
    var_12 = module_0.ScalarToken(var_2, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value1'
    var_6 = 6
    var_7 = 7
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #42
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = ''
    var_6 = module_0.ScalarToken(var_4, var_1, var_1, var_5)
    var_7 = 1
    var_8 = (var_6, var_7)
    var_9 = lambda s, end: var_8
    var_10 = {}
    var_11 = ''
    var_12 = module_1._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_11)
    var_13 = bool(var_12 == ({}, 2))
    assert var_13 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


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

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, end: (ScalarToken(var_4, end, end + var_5, s), end + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_index_error_handling_in_whitespace_skipping. Retrieved 8/15 statements.


def test_case_0():
    var_0 = '{"key": value'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = 1
    var_5 = var_3 + var_4
    var_6 = 1
    var_7 = var_5 + var_6
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    assert var_4 == ''



# Parsed testcases at query #46
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True
    var_14 = var_4.string
    assert var_14 == 'null'



# Parsed testcases at query #47
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #48
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    assert var_5 == 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = var_4.start_index
    assert var_7 == 0
    var_8 = var_4.end_index
    assert var_8 == 3
    var_9 = var_4.string
    assert var_9 == 'null'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 6/15 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = 1
    var_3 = module_0.Position(var_2, var_1, var_1)
    var_4 = 10
    var_5 = module_0.Position(var_2, var_4, var_4)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_tokenize_json_with_valid_content. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 13/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 11/13 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 10/13 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = False
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 8
    var_3 = 'value1'
    var_4 = 6
    var_5 = 7
    var_6 = 'value2'
    var_7 = lambda s, e: (ScalarToken(var_3, e, e + var_4, s), e + var_5) if e == var_2 else (ScalarToken(var_6, e, e + var_4, s), e + var_5)
    var_8 = 0
    var_9 = (var_0, var_8)
    var_10 = False
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 'value'
    var_3 = 5
    var_4 = 6
    var_5 = lambda s, e: (ScalarToken(var_2, e, e + var_3, s), e + var_4)
    var_6 = 0
    var_7 = (var_0, var_6)
    var_8 = False
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 8
    var_6 = 'value1'
    var_7 = 6
    var_8 = 7
    var_9 = 'value2'
    var_10 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s), e + var_8) if e == var_5 else (ScalarToken(var_9, e, e + var_7, s), e + var_8)
    var_11 = module_0._TokenizingJSONObject(var_3, var_4, var_10, var_1, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = module_0._TokenizingJSONObject(var_3, var_4, var_8, var_1, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting value'

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = {}
    var_2 = 15
    var_3 = 'value'
    var_4 = 5
    var_5 = 6
    var_6 = False
    var_7 = (var_0, var_6)
    var_8 = len(var_0)



# Parsed testcases at query #52
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = ''
    assert var_3 == ''



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 6/15 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = 1
    var_3 = module_0.Position(var_2, var_1, var_1)
    var_4 = 10
    var_5 = module_0.Position(var_2, var_4, var_4)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


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
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = len(var_0)

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

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, i: (ScalarToken(var_4, i, i + var_5, s), i + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/24 statements.


def test_case_0():
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_18 = '"test"'

def test_case_0():
    var_0 = 'Context'
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
    var_14 = (var_12, var_13)
    var_15 = lambda x, y, z: var_14
    var_16 = False
    var_17 = {}
    var_18 = '{"key": "value"}'

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'parse_string'
    var_4 = 'parse_float'
    var_5 = 'parse_int'
    var_6 = 'strict'
    var_7 = 'memo'
    var_8 = []
    var_9 = 3
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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
    var_0 = 'Context'
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



# Parsed testcases at query #56
#--------------------------




import typesystem.tokenize.tokens as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = '{"key": null}'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = 1
    var_8 = module_1.Position(var_7, var_7, var_1)
    var_9 = var_4.start
    var_10 = bool(var_4.start == var_8)
    assert var_10 is True
    var_11 = module_1.Position(var_7, var_5, var_2)
    var_12 = var_4.end
    var_13 = bool(var_4.end == var_11)
    assert var_13 is True
    var_14 = var_4.string
    assert var_14 == 'null'



# Parsed testcases at query #57
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 4
    assert var_5 == 4
    var_6 = var_4.value
    assert var_6 is None
    var_7 = var_4.start._index
    assert var_7 == 0
    var_8 = var_4.end._index
    assert var_8 == 3



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structures. Retrieved 11/12 statements.
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

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
    var_8 = bool(var_7 == [1, 2])
    assert var_8 is True
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_9, var_10]
    var_12 = var_1.lookup(var_11)
    var_13 = var_12.value
    assert var_13 == 3

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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_tokenize_json_with_valid_json. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.string
    var_3 = bool(var_1.string == var_0)
    assert var_3 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_line_32_predicate_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'nulx'
    var_1 = 0



