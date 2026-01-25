####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 14/25 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 19/30 statements.
# Partially parsed test_TokenizingJSONObject_whitespace_around_colon. Retrieved 14/23 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon_raises_error. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma_raises_error. Retrieved 7/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes_raises_error. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_scan_once_stop_iteration_raises_error. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 10
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 6
    var_14 = module_0.ScalarToken(var_5, var_13, var_13, var_0)
    var_15 = 14
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_14, var_12: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key" : "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 10
    var_10 = 16
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 13/18 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 14/19 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 14/19 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 9/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 9/15 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 10/15 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 14/18 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_string'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'test'
    var_6 = 6
    var_7 = lambda s, idx, strict: (var_5, idx + var_6)
    var_8 = True
    var_9 = {}
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = '"test"'
    var_13 = 0

def test_case_0():
    var_0 = {}
    var_1 = 1
    var_2 = 2
    var_3 = lambda string_idx, strict, scan, memo, content: (var_0, string_idx[var_1] + var_2)
    var_4 = 'Context'
    var_5 = ()
    var_6 = 'parse_object'
    var_7 = 'strict'
    var_8 = 'memo'
    var_9 = True
    var_10 = {}
    var_11 = {var_6: var_3, var_7: var_9, var_8: var_10}
    var_12 = [var_4, var_5, var_11]
    var_13 = '{}'
    var_14 = 0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = lambda string_idx, scan: (var_0, string_idx[var_1] + var_2)
    var_4 = 'Context'
    var_5 = ()
    var_6 = 'parse_array'
    var_7 = 'strict'
    var_8 = 'memo'
    var_9 = True
    var_10 = {}
    var_11 = {var_6: var_3, var_7: var_9, var_8: var_10}
    var_12 = [var_4, var_5, var_11]
    var_13 = '[]'
    var_14 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'null'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'true'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'false'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_int'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '123'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_float'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '123.45'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'invalid'
    var_9 = 'invalid'
    var_10 = 0
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 1
    var_3 = 2
    var_4 = lambda string_idx, strict, scan, memo, content: (var_1, string_idx[var_2] + var_3)
    var_5 = 'Context'
    var_6 = ()
    var_7 = 'parse_object'
    var_8 = 'strict'
    var_9 = 'memo'
    var_10 = True
    var_11 = {var_7: var_4, var_8: var_10, var_9: var_0}
    var_12 = [var_5, var_6, var_11]
    var_13 = '{}'
    var_14 = 0
    var_15 = bool(var_0 == {})
    assert var_15 is True



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/24 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/13 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 2
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 6
    var_9 = module_0.ScalarToken(var_4, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 9
    var_12 = 10
    var_13 = module_0.ScalarToken(var_10, var_11, var_12, var_0)
    var_14 = 13
    var_15 = module_0.ScalarToken(var_6, var_14, var_14, var_0)
    var_16 = {var_7: var_9, var_13: var_15}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 13
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_nextchar_not_comma_raises_error. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '{"key": "value" "another": "value2"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 13/18 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 9/17 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 7/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 7/12 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 7/13 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 15/19 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_string'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'test'
    var_6 = 4
    var_7 = lambda s, i, strict: (var_5, i + var_6)
    var_8 = True
    var_9 = {}
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = '"test"'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_object'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '{"key":"value"}'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'memo'
    var_4 = {}
    var_5 = '[1,2]'
    var_6 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'memo'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'memo'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'true'
    var_7 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'memo'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'false'
    var_7 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_int'
    var_3 = 'memo'
    var_4 = {}
    var_5 = '42'
    var_6 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_float'
    var_3 = 'memo'
    var_4 = {}
    var_5 = '3.14'
    var_6 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_string'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'test'
    var_6 = 4
    var_7 = lambda s, i, strict: (var_5, i + var_6)
    var_8 = True
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_7, var_3: var_8, var_4: var_11}
    var_13 = [var_0, var_1, var_12]
    var_14 = '"test"'
    var_15 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'memo'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'invalid'
    var_7 = 'invalid'
    var_8 = 0
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_24_evaluates_to_true. Retrieved 5/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = json.JSONDecoder().scan_once
    var_3 = '{"key": "value"}'
    var_4 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_53_evaluates_to_false. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = len(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_json_bytes. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_simple_object. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_simple_array. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_number. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_nested_object. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_nested_array. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_token_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_token_lookup_key. Retrieved 5/6 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'

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
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'

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
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"a": {"b": 1}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': 1}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": {"b": 1}}'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '[[1, 2], [3, 4]]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [[1, 2], [3, 4]])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[[1, 2], [3, 4]]'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.start.char_index
    assert var_2 == 0
    var_3 = len(var_0)
    var_4 = 1
    var_5 = var_3 - var_4
    var_6 = var_1.end.char_index
    var_7 = bool(var_1.end.char_index == var_5)
    assert var_7 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_dict. Retrieved 4/19 statements.
# Partially parsed test_make_scanner_list. Retrieved 4/24 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 4/16 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 4/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"test"'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'true'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'false'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '42'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '3.14'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '["item"]'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = ''
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '"test"'
    var_5 = 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_TokenizingJSONObject. Retrieved 1/11 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 5/16 statements.


def test_case_0():
    var_0 = json.JSONDecoder().scan_once
    var_1 = True
    var_2 = {}
    var_3 = '{"key": "value"}'
    var_4 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_list. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_list_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_dict. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_dict_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 17/18 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
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
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, "two", false]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two', False])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, "two", false]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 18
    var_11 = 17
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value", "num": 42}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value', 'num': 42})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value", "num": 42}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 28
    var_11 = 27
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"list": [1, 2, 3], "nested": {"inner": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'inner'
    var_9 = True
    var_10 = {var_8: var_9}
    var_11 = {var_2: var_7, var_3: var_10}
    var_12 = var_1.value
    var_13 = bool(var_1.value == var_11)
    assert var_13 is True
    var_14 = var_1.string
    assert var_14 == '{"list": [1, 2, 3], "nested": {"inner": true}}'
    var_15 = 0
    var_16 = module_1.Position(var_9, var_9, var_15)
    var_17 = var_1.start
    var_18 = bool(var_1.start == var_16)
    assert var_18 is True
    var_19 = 48
    var_20 = 47
    var_21 = module_1.Position(var_9, var_19, var_20)
    var_22 = var_1.end
    var_23 = bool(var_1.end == var_21)
    assert var_23 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'"hello"'
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

def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "key": "value"\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{\n  "key": "value"\n}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 20
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_32_false. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'x'
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nextchar_not_double_quote_raises_error. Retrieved 11/29 statements.


def test_case_0():
    var_0 = '{"key": "value", "another": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = len(var_0)
    var_6 = '{"key": "value", invalid: 123}'
    var_7 = 0
    var_8 = (var_6, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_invalid_bytes. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n\t  '
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
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, "two", false]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 'two', False])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, "two", false]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 18
    var_11 = 17
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [{"b": 2}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [{'b': 2}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [{"b": 2}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 18
    var_11 = 17
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"unclosed": '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b'{"invalid": \x80}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'invalid'
    var_3 = bool('invalid' in var_1.value)
    assert var_3 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "key": "value"\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{\n  "key": "value"\n}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 20
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 13/18 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 9/17 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 7/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 9/14 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 9/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 9/15 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 10/15 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_string'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = 'test'
    var_6 = 6
    var_7 = lambda s, i, strict: (var_5, i + var_6)
    var_8 = True
    var_9 = {}
    var_10 = {var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = '"test"'
    var_13 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_object'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '{"key":"value"}'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_array'
    var_3 = 'memo'
    var_4 = {}
    var_5 = '[1,2]'
    var_6 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'null'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'true'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'false'
    var_9 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_int'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '123'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'parse_float'
    var_3 = 'strict'
    var_4 = 'memo'
    var_5 = True
    var_6 = {}
    var_7 = '123.45'
    var_8 = 0

def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'memo'
    var_4 = True
    var_5 = {}
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'invalid'
    var_9 = 'invalid'
    var_10 = 0
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Context'
    var_2 = ()
    var_3 = 'parse_string'
    var_4 = 'strict'
    var_5 = 'memo'
    var_6 = 'test'
    var_7 = 6
    var_8 = lambda s, i, strict: (var_6, i + var_7)
    var_9 = True
    var_10 = {var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_1, var_2, var_10]
    var_12 = '"test"'
    var_13 = 0
    var_14 = bool(var_0 == {})
    assert var_14 is True



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_dict. Retrieved 6/16 statements.
# Partially parsed test_make_scanner_list. Retrieved 6/13 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 6/11 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 7/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = {}
    var_4 = '"test"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '42'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '3.14'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = lambda self, string_idx, scan_once: ([], 2)
    var_3 = {}
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '"test"'
    var_7 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = {}
    var_4 = ''
    var_5 = ''
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 20/29 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 6
    var_10 = 7
    var_11 = module_0.ScalarToken(var_5, var_9, var_10, var_0)
    var_12 = 'b'
    var_13 = 10
    var_14 = 11
    var_15 = module_0.ScalarToken(var_12, var_13, var_14, var_0)
    var_16 = 14
    var_17 = 15
    var_18 = module_0.ScalarToken(var_7, var_16, var_17, var_0)
    var_19 = {var_8: var_11, var_15: var_18}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 13
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 7
    var_10 = 9
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 5
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 9
    var_13 = 10
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 9
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 13
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 'key'
    var_13 = bool('key' in var_1)
    assert var_13 is True
    var_14 = var_1['key']
    assert var_14 == 'key'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 8/29 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = '{"key": "value"}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = 0
    var_7 = len(var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 13/19 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 18/29 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/20 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 7/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    assert var_3 == 2
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 6
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 9
    var_13 = 10
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 14
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 10
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: 123}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_array_with_elements. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_object_with_key_value. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
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
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[1, true, null]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, True, None])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, true, null]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [1, 2], "b": {"c": true}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': True}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [1, 2], "b": {"c": true}}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 33
    var_11 = 32
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 16/24 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/15 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = module_0.ScalarToken(var_6, var_5, var_5, var_0)
    var_8 = 6
    var_9 = module_0.ScalarToken(var_5, var_8, var_8, var_0)
    var_10 = 'b'
    var_11 = 10
    var_12 = module_0.ScalarToken(var_10, var_11, var_11, var_0)
    var_13 = 2
    var_14 = 13
    var_15 = module_0.ScalarToken(var_13, var_14, var_14, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 4
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 7
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 11
    var_13 = 12
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 15
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_10, var_14: var_16}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 13
    var_11 = 19
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/22 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 7
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 2
    var_15 = 14
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_scanner_with_null. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_number. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 6/18 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 6/14 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid. Retrieved 6/14 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = 'null'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = 'true'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = 'false'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = '123'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = '12.34'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('parsed', idx + 8)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = '"string"'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = lambda self, s, idx, strict: ('', idx)
    var_5 = lambda self, args, scan: ([], args[1])
    var_6 = 'null'
    var_7 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, s, idx, strict: ('', idx)
    var_3 = lambda self, args, scan: ([], args[1])
    var_4 = 'invalid'
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 7
    var_10 = 9
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 5
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 9
    var_13 = 10
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 9
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 13
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 'key'
    var_13 = bool('key' in var_1)
    assert var_13 is True
    var_14 = var_1['key']
    assert var_14 == 'key'

def test_case_0():
    var_0 = '{"key":'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 14/20 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 19/31 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 15/21 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/15 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 14/20 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 123
    var_9 = 8
    var_10 = 10
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    assert var_2 == 2
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = 'a'
    var_7 = 2
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 8
    var_11 = 9
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 6
    var_14 = module_0.ScalarToken(var_5, var_13, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_14, var_12: var_16}
    var_18 = len(var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : 123 }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 10
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = len(var_0)

def test_case_0():
    var_0 = '{"key" 123}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{123: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 4
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 13
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}
    var_13 = 'key'
    var_14 = bool('key' in var_1)
    assert var_14 is True
    var_15 = var_1['key']
    assert var_15 == 'key'
    var_16 = len(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 7/18 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 0
    var_4 = None
    var_5 = 3
    var_6 = module_0.ScalarToken(var_4, var_3, var_5, var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_true. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = len(var_0)



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_scanner_parse_object_is_not_TokenizingJSONObject. Retrieved 1/5 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/11 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 14/22 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 17/28 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 15/23 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 6/12 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 7/19 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 6/12 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 8
    var_10 = 12
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 0

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = module_0.ScalarToken(var_6, var_5, var_5, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = module_0.ScalarToken(var_8, var_9, var_9, var_0)
    var_11 = 6
    var_12 = module_0.ScalarToken(var_5, var_11, var_11, var_0)
    var_13 = 2
    var_14 = 13
    var_15 = module_0.ScalarToken(var_13, var_14, var_14, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 4
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 10
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 0

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_null_token_creation. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'Context'
    var_1 = ()
    var_2 = 'strict'
    var_3 = 'parse_float'
    var_4 = 'parse_int'
    var_5 = 'memo'
    var_6 = True
    var_7 = {}
    var_8 = 'null'
    var_9 = 0



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_with_empty_string. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_null. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_number. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 5/18 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 5/20 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 5/18 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = ''
    var_3 = ''
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'null'
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'true'
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'false'
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '42'
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '3.14'
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"hello"'
    var_3 = '"hello"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '{}'
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '[]'
    var_3 = '[]'
    var_4 = 0



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_scalar_token_null_value. Retrieved 5/6 statements.


import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_object. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_array. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid. Retrieved 5/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = '[]'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '"test"'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = 'invalid'
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 12/16 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 7
    var_2 = ' \t\n\r'
    var_3 = 'Match'
    var_4 = ()
    var_5 = 'end'
    var_6 = lambda s, idx: type(var_3, var_4, {var_5: lambda : idx})()
    var_7 = 1
    var_8 = var_1 + var_7
    var_9 = 1
    var_10 = var_8 + var_9
    var_11 = var_6(var_0, var_10)
    var_12 = var_0[var_8]
    var_13 = bool(var_0[var_8] not in var_2)
    assert var_13 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_false. Retrieved 7/18 statements.


def test_case_0():
    var_0 = lambda self, *args: ([], 0)
    var_1 = lambda self, *args: ('', 0)
    var_2 = True
    var_3 = {}
    var_4 = ''
    var_5 = 'x'
    var_6 = 0



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_simple_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 14/15 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_negative_number. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_exponential_number. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   \n\t  '
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
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [1, {"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = 'b'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_2: var_7}
    var_9 = var_1.value
    var_10 = bool(var_1.value == var_8)
    assert var_10 is True
    var_11 = var_1.string
    assert var_11 == '{"a": [1, {"b": true}]}'
    var_12 = 0
    var_13 = module_1.Position(var_5, var_5, var_12)
    var_14 = var_1.start
    var_15 = bool(var_1.start == var_13)
    assert var_15 is True
    var_16 = 24
    var_17 = 23
    var_18 = module_1.Position(var_5, var_16, var_17)
    var_19 = var_1.end
    var_20 = bool(var_1.end == var_18)
    assert var_20 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"unclosed": '
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{\n  "key": "value"\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{\n  "key": "value"\n}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 20
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '-42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == -42
    var_3 = var_1.string
    assert var_3 == '-42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 2
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '1.23e-4'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 0.000123)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '1.23e-4'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 7
    var_11 = 6
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/21 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/22 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = '{}'
    var_3 = 0
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None
    var_7 = lambda s, idx: (ScalarToken(var_6, idx, idx, var_0), idx + var_5)
    var_8 = {}

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'value'
    var_9 = 9
    var_10 = 15
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 3
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 9
    var_10 = 11
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)

import typesystem.tokenize.tokens as module_0

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 11
    var_11 = 17
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)

def test_case_0():
    var_0 = '{"key" "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ':' delimiter"

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting property name enclosed in double quotes'

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_object. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_object. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_empty_array. Retrieved 7/8 statements.
# Partially parsed test_tokenize_json_array. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '   '
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
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 16
    var_11 = 15
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = module_1.Position(var_5, var_10, var_5)
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

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '{"a": [1, {"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, {'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [1, {"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 24
    var_11 = 23
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = b'{"test": 123}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'test': 123})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"test": 123}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 13
    var_11 = 12
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True

import typesystem.tokenize.tokenize_json as module_0

def test_case_0():
    var_0 = '{"invalid": }'
    var_1 = module_0.tokenize_json(var_0)

import typesystem.tokenize.tokenize_json as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = '[\n  1,\n  2\n]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[\n  1,\n  2\n]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 10
    var_12 = module_1.Position(var_10, var_5, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



