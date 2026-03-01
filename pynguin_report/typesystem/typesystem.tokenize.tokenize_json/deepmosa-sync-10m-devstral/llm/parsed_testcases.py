####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/11 statements.


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
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = len(var_0)

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
    var_4 = 9
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

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



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = 1
    var_3 = var_1 + var_2
    var_4 = var_0[var_1:var_3]
    var_5 = ' \t\n\r'
    var_6 = bool(var_4 != '"' and var_4 not in var_5 and (var_4 != '}'))
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = '  '
    var_1 = 0
    var_2 = ' \t\n\r'
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_1:var_4]
    var_6 = bool(var_5 in var_2)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_values. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 10/12 statements.


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
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 8
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s) if e == var_4 else ScalarToken(var_7, e, e + var_6, s), e + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = len(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
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
    var_3 = False
    var_4 = 8
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s) if e == var_4 else ScalarToken(var_7, e, e + var_6, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
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
    var_3 = False
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 6/7 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 12/14 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}

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

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = '{"key1": "value1", "key2": "value2"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 8
    var_6 = 'value1'
    var_7 = 6
    var_8 = 'value2'
    var_9 = 7
    var_10 = lambda s, e: (ScalarToken(var_6, e, e + var_7, s) if e == var_5 else ScalarToken(var_8, e, e + var_7, s), e + var_9)
    var_11 = {}

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = '{"key" : "value"}'
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = {}

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
    var_4 = 8
    var_5 = 'value1'
    var_6 = 6
    var_7 = 'value2'
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s) if e == var_4 else ScalarToken(var_7, e, e + var_6, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/15 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 2/16 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = '{"key": "test"}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": true}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": false}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": {}}'
    var_1 = 7

def test_case_0():
    var_0 = '{"key": []}'
    var_1 = 7



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = '"test"'
    var_1 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0

def test_case_0():
    var_0 = '["item1", "item2"]'
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



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = '  '
    var_1 = 0
    var_2 = 'Match'
    var_3 = ()
    var_4 = 'end'
    var_5 = 2
    var_6 = lambda s, end: type(var_2, var_3, {var_4: lambda self: end + var_5})()
    var_7 = ' '
    var_8 = var_0[var_1]
    var_9 = bool(var_0[var_1] in var_7)
    assert var_9 is True
    var_10 = 1
    var_11 = var_1 + var_10
    var_12 = var_0[var_11]
    var_13 = bool(var_0[var_11] in var_7)
    assert var_13 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 19/20 statements.
# Partially parsed test__TokenizingJSONObject_whitespace_handling. Retrieved 14/15 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, var_3, var_3, s), e)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 8
    var_4 = 13
    var_5 = 14
    var_6 = lambda s, e: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_12 = {var_10: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = {}
    var_2 = 9
    var_3 = 'value1'
    var_4 = 15
    var_5 = 'value2'
    var_6 = 25
    var_7 = 31
    var_8 = 32
    var_9 = lambda s, e: ScalarToken(var_3, var_2, var_4, s) if e == var_2 else (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = 0
    var_11 = (var_0, var_10)
    var_12 = False
    var_13 = 'key1'
    var_14 = 'key2'
    var_15 = module_0.ScalarToken(var_3, var_2, var_4, var_0)
    var_16 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_17 = {var_13: var_15, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = {}
    var_2 = 'value'
    var_3 = 10
    var_4 = 15
    var_5 = 16
    var_6 = lambda s, e: (ScalarToken(var_2, var_3, var_4, s), var_5)
    var_7 = 0
    var_8 = (var_0, var_7)
    var_9 = False
    var_10 = 'key'
    var_11 = module_0.ScalarToken(var_2, var_3, var_4, var_0)
    var_12 = {var_10: var_11}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 8
    var_7 = 13
    var_8 = 14
    var_9 = lambda s, e: (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_9, var_1, var_0)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 9
    var_6 = 'value1'
    var_7 = 15
    var_8 = 'value2'
    var_9 = 25
    var_10 = 31
    var_11 = 32
    var_12 = lambda s, e: ScalarToken(var_6, var_5, var_7, s) if e == var_5 else (ScalarToken(var_8, var_9, var_10, s), var_11)
    var_13 = module_0._TokenizingJSONObject(var_3, var_4, var_12, var_1, var_0)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 8
    var_7 = 13
    var_8 = 14
    var_9 = lambda s, e: (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_9, var_1, var_0)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = False
    var_5 = 'value'
    var_6 = 7
    var_7 = 12
    var_8 = 13
    var_9 = lambda s, e: (ScalarToken(var_5, var_6, var_7, s), var_8)
    var_10 = module_0._TokenizingJSONObject(var_3, var_4, var_9, var_1, var_0)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 17/22 statements.
# Partially parsed test_make_scanner_scans_string_token. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_scans_dict_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_list_token. Retrieved 20/26 statements.
# Partially parsed test_make_scanner_scans_null_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_true_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_false_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_number_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_scans_float_token. Retrieved 18/24 statements.
# Partially parsed test_make_scanner_raises_stop_iteration. Retrieved 19/25 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 19/24 statements.


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
    var_4 = 'strict'
    var_5 = 'parse_float'
    var_6 = 'parse_int'
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

import typesystem.tokenize.tokens as module_0

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
    var_14 = lambda x, y: (var_12, y + var_13)
    var_15 = (var_10, var_9)
    var_16 = lambda x, y, z: var_15
    var_17 = False
    var_18 = {}
    var_19 = '[1, 2, 3]'

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
    var_16 = 'test'
    var_17 = 'value'
    var_18 = {var_16: var_17}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 5/14 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = None
    var_3 = 10
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenizing_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenizing_single_key_value_pair. Retrieved 17/18 statements.
# Partially parsed test_tokenizing_multiple_key_value_pairs. Retrieved 26/27 statements.
# Partially parsed test_tokenizing_with_whitespace. Retrieved 17/18 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
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
    var_13 = 14
    var_14 = (var_8, var_13)
    var_15 = lambda s, e: var_14
    var_16 = {}

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
    var_17 = '{"key1": "value1", "key2": "value2"}'
    var_18 = 0
    var_19 = (var_17, var_18)
    var_20 = True
    var_21 = 8
    var_22 = 16
    var_23 = 34
    var_24 = lambda s, e: (var_8 if e == var_21 else var_16, var_22 if e == var_21 else var_23)
    var_25 = {}

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
    var_13 = 16
    var_14 = (var_8, var_13)
    var_15 = lambda s, e: var_14
    var_16 = {}

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key": "value",}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 8
    var_7 = 13
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key": "value",}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = 14
    var_14 = (var_8, var_13)
    var_15 = lambda s, e: var_14
    var_16 = {}
    var_17 = module_1._TokenizingJSONObject(var_11, var_12, var_15, var_16, var_0)
    var_18 = bool(False)
    assert var_18 is True

import typesystem.tokenize.tokens as module_0
import typesystem.tokenize.tokenize_json as module_1

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 'key'
    var_2 = 1
    var_3 = 4
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 'value'
    var_6 = 7
    var_7 = 12
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = '{"key" "value"}'
    var_10 = 0
    var_11 = (var_9, var_10)
    var_12 = True
    var_13 = 13
    var_14 = (var_8, var_13)
    var_15 = lambda s, e: var_14
    var_16 = {}
    var_17 = module_1._TokenizingJSONObject(var_11, var_12, var_15, var_16, var_0)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #14
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
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
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
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
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
    var_5 = 4
    var_6 = 5
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
    var_4 = 8
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = '  '
    var_1 = 0
    var_2 = var_0[var_1]
    var_3 = 1
    var_4 = var_1 + var_3
    var_5 = var_0[var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 6/8 statements.
# Partially parsed test_tokenizing_json_object_multiple_pairs. Retrieved 7/9 statements.
# Partially parsed test_tokenizing_json_object_with_whitespace. Retrieved 6/8 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 2
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}

def test_case_0():
    var_0 = '{ "a" : 1 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a" 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = "Expecting ':' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 2
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e + var_3)
    var_6 = {}
    var_7 = module_0._TokenizingJSONObject(var_2, var_3, var_5, var_6, var_0)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "Expecting ',' delimiter"

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": 1'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Expecting value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{1: 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, e: (ScalarToken(var_3, e, e, s), e + var_3)
    var_5 = {}
    var_6 = module_0._TokenizingJSONObject(var_2, var_3, var_4, var_5, var_0)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Expecting property name enclosed in double quotes'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/14 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 3/17 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/14 statements.


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
    var_1 = '['
    var_2 = 0

def test_case_0():
    var_0 = '{}'
    var_1 = '{'
    var_2 = 0

def test_case_0():
    var_0 = ''
    var_1 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 5/6 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 10/11 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 10/11 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 16/17 statements.


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
    var_0 = '{"a": 1}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, i: (ScalarToken(var_3, i, i, s), i + var_3)
    var_5 = {}
    var_6 = 'a'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_3, var_7, var_7, var_0)
    var_9 = {var_6: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 2
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 3
    var_10 = module_0.ScalarToken(var_3, var_9, var_9, var_0)
    var_11 = 11
    var_12 = module_0.ScalarToken(var_4, var_11, var_11, var_0)
    var_13 = {var_7: var_10, var_8: var_12}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "a" : 1 }'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = lambda s, i: (ScalarToken(var_3, i, i, s), i + var_3)
    var_5 = {}
    var_6 = 'a'
    var_7 = 6
    var_8 = module_0.ScalarToken(var_3, var_7, var_7, var_0)
    var_9 = {var_6: var_8}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": {"b": 2}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 2
    var_5 = lambda s, i: (ScalarToken(var_4, i, i, s), i + var_3)
    var_6 = {}
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 10
    var_10 = module_0.ScalarToken(var_4, var_9, var_9, var_0)
    var_11 = {var_8: var_10}
    var_12 = 3
    var_13 = 13
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = {var_7: var_14}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 7/8 statements.
# Partially parsed test_tokenizing_json_object_single_pair. Retrieved 10/12 statements.
# Partially parsed test_tokenizing_json_object_multiple_pairs. Retrieved 12/14 statements.
# Partially parsed test_tokenizing_json_object_with_whitespace. Retrieved 12/14 statements.


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
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
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
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = len(var_0)

def test_case_0():
    var_0 = '  { "key" : "value" , "key2" : "value2" }  '
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 10
    var_5 = 'value'
    var_6 = 'value2'
    var_7 = 5
    var_8 = 6
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = len(var_0)

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
    var_4 = 10
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 6
    var_8 = 7
    var_9 = lambda s, e: (ScalarToken(var_5 if e == var_4 else var_6, e, e + var_7, s), e + var_8)
    var_10 = {}
    var_11 = module_0._TokenizingJSONObject(var_2, var_3, var_9, var_10, var_0)
    var_12 = bool(False)
    assert var_12 is True

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_array_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_object_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_empty_string. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test content'

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
    var_0 = '123.456'
    var_1 = 0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0

def test_case_0():
    var_0 = '""'
    var_1 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 20/21 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 15/16 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 17/18 statements.


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
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 8
    var_11 = 12
    var_12 = module_0.ScalarToken(var_4, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key1'
    var_10 = 'key2'
    var_11 = 8
    var_12 = 13
    var_13 = module_0.ScalarToken(var_4, var_11, var_12, var_0)
    var_14 = 'value2'
    var_15 = 23
    var_16 = 28
    var_17 = module_0.ScalarToken(var_14, var_15, var_16, var_0)
    var_18 = {var_9: var_13, var_10: var_17}
    var_19 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_4, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"outer": {"inner": "value"}}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 4
    var_6 = 5
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'outer'
    var_10 = 'inner'
    var_11 = 18
    var_12 = 22
    var_13 = module_0.ScalarToken(var_4, var_11, var_12, var_0)
    var_14 = {var_10: var_13}
    var_15 = {var_9: var_14}
    var_16 = len(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 7/11 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test content'

import typesystem.base as module_0

def test_case_0():
    var_0 = '"test"'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 6
    var_5 = 5
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'null'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 5
    var_5 = 3
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'true'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 5
    var_5 = 3
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = 'false'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 6
    var_5 = 4
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = '123'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 4
    var_5 = 2
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = '123.45'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 7
    var_5 = 5
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = '[1, 2, 3]'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 9
    var_5 = 8
    var_6 = module_0.Position(var_2, var_4, var_5)

import typesystem.base as module_0

def test_case_0():
    var_0 = '{"a": 1}'
    var_1 = 0
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_1)
    var_4 = 8
    var_5 = 7
    var_6 = module_0.Position(var_2, var_4, var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 5/8 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = 7
    var_2 = None
    var_3 = 10
    var_4 = module_0.ScalarToken(var_2, var_1, var_3, var_0)



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '{"key":  value}'
    var_1 = 7
    var_2 = var_0[var_1]
    assert var_2 == ' '
    var_3 = 1
    var_4 = var_1 + var_3
    assert var_4 == 8
    var_5 = var_0[var_4]
    assert var_5 == ' '



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_json_list. Retrieved 20/21 statements.
# Partially parsed test_tokenize_json_dict. Retrieved 22/23 statements.
# Partially parsed test_tokenize_json_nested. Retrieved 16/17 statements.
# Partially parsed test_tokenize_json_bytes. Retrieved 6/7 statements.


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
    var_17 = var_1.lookup(var_16)
    var_18 = var_17.value
    assert var_18 == 1
    var_19 = 'b'
    var_20 = [var_19]
    var_21 = var_1.lookup(var_20)
    var_22 = var_21.value
    assert var_22 == 2
    var_23 = [var_15]
    var_24 = var_1.lookup_key(var_23)
    var_25 = var_24.value
    assert var_25 == 'a'
    var_26 = [var_19]
    var_27 = var_1.lookup_key(var_26)
    var_28 = var_27.value
    assert var_28 == 'b'

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
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a":}'
    var_1 = module_0.tokenize_json(var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 14/15 statements.
# Partially parsed test__TokenizingJSONObject_empty_key. Retrieved 15/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = lambda s, e: (ScalarToken(var_4, e, e, s), e)
    var_6 = {}

import typesystem.tokenize.tokens as module_0

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
    var_10 = 10
    var_11 = module_0.ScalarToken(var_4, var_6, var_10, var_0)
    var_12 = {var_9: var_11}
    var_13 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = 'key1'
    var_10 = 12
    var_11 = module_0.ScalarToken(var_4, var_6, var_10, var_0)
    var_12 = {var_9: var_11}
    var_13 = len(var_0)

import typesystem.tokenize.tokens as module_0

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
    var_10 = 10
    var_11 = module_0.ScalarToken(var_4, var_6, var_10, var_0)
    var_12 = {var_9: var_11}
    var_13 = len(var_0)

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

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key1": "value1" "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value1'
    var_5 = 6
    var_6 = 7
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = module_0._TokenizingJSONObject(var_2, var_3, var_7, var_8, var_0)

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

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"": "value"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = ''
    var_10 = 3
    var_11 = 7
    var_12 = module_0.ScalarToken(var_4, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_False. Retrieved 13/17 statements.


import re as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = 1
    var_3 = var_1 - var_2
    var_4 = '\\s'
    var_5 = module_0.compile(var_4)
    var_6 = var_5.match
    var_7 = ' \t\n\r'
    var_8 = 1
    var_9 = var_3 + var_8
    var_10 = 1
    var_11 = var_9 + var_10
    var_12 = var_6(var_0, var_11)
    var_13 = bool(True)
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_key_value_pair. Retrieved 10/12 statements.
# Partially parsed test__TokenizingJSONObject_multiple_key_value_pairs. Retrieved 6/12 statements.
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
    var_3 = True
    var_4 = 'value'
    var_5 = 5
    var_6 = 6
    var_7 = lambda s, e: (ScalarToken(var_4, e, e + var_5, s), e + var_6)
    var_8 = {}
    var_9 = len(var_0)

def test_case_0():
    var_0 = '{"key1": "value1", "key2": "value2"}'
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = len(var_0)

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
    var_9 = len(var_0)

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
    var_7 = True
    var_8 = len(var_0)
    var_9 = bool(var_1 == var_8)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 3/12 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 2/8 statements.
# Partially parsed test_make_scanner_with_integer_token. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test content'

def test_case_0():
    var_0 = 'test'
    var_1 = 6
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = len(var_0)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 8
    var_5 = '[1, 2, 3]'
    var_6 = 0

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

def test_case_0():
    var_0 = 'null'
    var_1 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_with_string_token. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 2/6 statements.
# Partially parsed test_make_scanner_with_integer_token. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 2/7 statements.
# Partially parsed test_make_scanner_clears_memo. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test content'

def test_case_0():
    var_0 = 'test'
    var_1 = 6
    var_2 = '{"key": "test"}'
    var_3 = 1

def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = len(var_2)
    var_5 = 1
    var_6 = var_4 - var_5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 8
    var_5 = '[1, 2, 3]'
    var_6 = 0
    var_7 = len(var_5)
    var_8 = var_7 - var_0

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
    var_0 = '42'
    var_1 = 0

def test_case_0():
    var_0 = '3.14'
    var_1 = 0

def test_case_0():
    var_0 = 'test'
    var_1 = 0



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = len(var_0)
    var_2 = var_0[var_1]
    var_3 = ''
    assert var_3 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_scanner_with_string_token. Retrieved 11/15 statements.
# Partially parsed test_make_scanner_with_dict_token. Retrieved 10/14 statements.
# Partially parsed test_make_scanner_with_list_token. Retrieved 14/18 statements.
# Partially parsed test_make_scanner_with_null_token. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_with_true_token. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_with_false_token. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_with_number_token. Retrieved 9/13 statements.
# Partially parsed test_make_scanner_with_float_token. Retrieved 9/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = 'test'
    var_5 = 5
    var_6 = lambda s, i, strict: (var_4, i + var_5)
    var_7 = False
    var_8 = {}
    var_9 = 'test content'
    var_10 = '"test"'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = 'key'
    var_5 = 3
    var_6 = lambda s, i, strict: (var_4, i + var_5)
    var_7 = False
    var_8 = {}
    var_9 = '{"key": "value"}'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = (var_3, var_4)
    var_6 = lambda x, y: var_5
    var_7 = None
    var_8 = 0
    var_9 = (var_7, var_8)
    var_10 = lambda s, i, strict: var_9
    var_11 = False
    var_12 = {}
    var_13 = '[1, 2, 3]'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda s, i, strict: var_4
    var_6 = False
    var_7 = {}
    var_8 = 'null'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda s, i, strict: var_4
    var_6 = False
    var_7 = {}
    var_8 = 'true'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda s, i, strict: var_4
    var_6 = False
    var_7 = {}
    var_8 = 'false'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda s, i, strict: var_4
    var_6 = False
    var_7 = {}
    var_8 = '123'

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x, y: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda s, i, strict: var_4
    var_6 = False
    var_7 = {}
    var_8 = '123.45'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_scan_once_raises_stop_iteration. Retrieved 6/12 statements.


def test_case_0():
    var_0 = '{"key":'
    var_1 = 6
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = {}
    var_5 = '{"key":}'



# Parsed testcases at query #22
#--------------------------




import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": null}'
    var_1 = None
    var_2 = 7
    var_3 = 10
    var_4 = module_0.ScalarToken(var_1, var_2, var_3, var_0)
    var_5 = 11
    var_6 = var_4.string
    assert var_6 == 'null'
    var_7 = var_4.value
    assert var_7 is None
    var_8 = var_4.start.line_no
    assert var_8 == 1
    var_9 = var_4.start.column_no
    assert var_9 == 8
    var_10 = var_4.end.line_no
    assert var_10 == 1
    var_11 = var_4.end.column_no
    assert var_11 == 11



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/15 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_int. Retrieved 2/16 statements.
# Partially parsed test_make_scanner_scans_number_float. Retrieved 2/16 statements.
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test__TokenizingJSONObject_single_pair. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_multiple_pairs. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_with_whitespace. Retrieved 16/17 statements.
# Partially parsed test__TokenizingJSONObject_nested_object. Retrieved 19/20 statements.


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
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
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
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = 'key1'
    var_10 = 8
    var_11 = 13
    var_12 = module_0.ScalarToken(var_5, var_10, var_11, var_0)
    var_13 = {var_9: var_12}
    var_14 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '  { "key" : "value" }  '
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'value'
    var_6 = 5
    var_7 = 6
    var_8 = lambda s, e: (ScalarToken(var_5, e, e + var_6, s), e + var_7)
    var_9 = 'key'
    var_10 = 13
    var_11 = 17
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
    var_8 = 17
    var_9 = 18
    var_10 = lambda s, e: (ScalarToken(var_7, e, e + var_8, s), e + var_9)
    var_11 = 'outer'
    var_12 = {var_5: var_6}
    var_13 = 8
    var_14 = 24
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = {var_11: var_15}
    var_17 = len(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_scanner_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_scanner_scans_string. Retrieved 4/9 statements.
# Partially parsed test_make_scanner_scans_object. Retrieved 7/17 statements.
# Partially parsed test_make_scanner_scans_array. Retrieved 9/19 statements.
# Partially parsed test_make_scanner_scans_null. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_scans_true. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_scans_false. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_scans_number. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_scans_float. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'key'
    var_4 = 5
    var_5 = '{"key": "value"}'
    var_6 = len(var_5)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = 'test'
    var_6 = 5
    var_7 = '[1, 2, 3]'
    var_8 = len(var_7)

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = 'null'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = 'true'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = 'false'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = '123'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = '123.45'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = 'test'
    var_4 = 5
    var_5 = 'invalid'
    var_6 = 0



