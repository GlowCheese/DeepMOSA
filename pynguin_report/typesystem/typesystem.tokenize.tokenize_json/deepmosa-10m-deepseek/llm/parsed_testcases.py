####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 4/17 statements.
# Partially parsed test_make_scanner_stop_iteration. Retrieved 4/16 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '"test"'
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
    var_2 = 'null'
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
    var_2 = '[1]'
    var_3 = 0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 0
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #3
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/10 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 8/19 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 15/27 statements.
# Partially parsed test_TokenizingJSONObject_whitespace_handling. Retrieved 7/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon_raises_error. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma_raises_error. Retrieved 7/15 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote_on_key_raises_error. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_raises_error. Retrieved 6/10 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = {}

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 0
    var_6 = len(var_0)

import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 10
    var_11 = 12
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = len(var_0)

def test_case_0():
    var_0 = ' { "key" : "value" } '
    var_1 = 1
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 0

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

def test_case_0():
    var_0 = '{"key": "value", "key": "other"}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_tokenizing_json_object_nextchar_not_comma. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_nextchar_is_closing_brace_after_processing_key_value_pair. Retrieved 15/20 statements.



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = {}
    var_3 = 'value'
    var_4 = 9
    var_5 = 15
    var_6 = module_0.ScalarToken(var_3, var_4, var_5, var_0)
    var_7 = 16
    var_8 = (var_6, var_7)
    var_9 = True
    var_10 = (var_0, var_1)
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_3, var_4, var_5, var_0)
    var_13 = {var_11: var_12}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True


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



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_json_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_integer. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_number_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_empty_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_list_with_elements. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_empty_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_dict_with_elements. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_nested_structure. Retrieved 11/12 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_multiline_string. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_lookup_in_dict. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_lookup_in_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_json_lookup_key. Retrieved 6/7 statements.


import typesystem.tokenize.tokenize_json as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 4
    var_9 = var_1.end.char_index
    assert var_9 == 3


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 4
    var_9 = var_1.end.char_index
    assert var_9 == 3


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 5
    var_9 = var_1.end.char_index
    assert var_9 == 4


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 2
    var_9 = var_1.end.char_index
    assert var_9 == 1


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.start.char_index
    assert var_7 == 0
    var_8 = var_1.end.line_no
    assert var_8 == 1
    var_9 = var_1.end.column_no
    assert var_9 == 4
    var_10 = var_1.end.char_index
    assert var_10 == 3


def test_case_0():
    var_0 = '"hello"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == '"hello"'
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 1
    var_8 = var_1.end.column_no
    assert var_8 == 7
    var_9 = var_1.end.char_index
    assert var_9 == 6


def test_case_0():
    var_0 = '[]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[]'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.start.char_index
    assert var_7 == 0
    var_8 = var_1.end.line_no
    assert var_8 == 1
    var_9 = var_1.end.column_no
    assert var_9 == 2
    var_10 = var_1.end.char_index
    assert var_10 == 1


def test_case_0():
    var_0 = '[1, true, null]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, True, None])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, true, null]'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.start.char_index
    assert var_7 == 0
    var_8 = var_1.end.line_no
    assert var_8 == 1
    var_9 = var_1.end.column_no
    assert var_9 == 16
    var_10 = var_1.end.char_index
    assert var_10 == 15


def test_case_0():
    var_0 = '{}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{}'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.start.char_index
    assert var_7 == 0
    var_8 = var_1.end.line_no
    assert var_8 == 1
    var_9 = var_1.end.column_no
    assert var_9 == 2
    var_10 = var_1.end.char_index
    assert var_10 == 1


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"key": "value"}'
    var_5 = var_1.start.line_no
    assert var_5 == 1
    var_6 = var_1.start.column_no
    assert var_6 == 1
    var_7 = var_1.start.char_index
    assert var_7 == 0
    var_8 = var_1.end.line_no
    assert var_8 == 1
    var_9 = var_1.end.column_no
    assert var_9 == 16
    var_10 = var_1.end.char_index
    assert var_10 == 15


def test_case_0():
    var_0 = '{"list": [1, 2], "nested": {"bool": false}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'list'
    var_3 = 'nested'
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 'bool'
    var_8 = False
    var_9 = {var_7: var_8}
    var_10 = {var_2: var_6, var_3: var_9}
    var_11 = var_1.value
    var_12 = bool(var_1.value == var_10)
    assert var_12 is True
    var_13 = var_1.start.line_no
    assert var_13 == 1
    var_14 = var_1.start.column_no
    assert var_14 == 1
    var_15 = var_1.start.char_index
    assert var_15 == 0


def test_case_0():
    var_0 = b'"bytes"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    assert var_2 == 'bytes'
    var_3 = var_1.string
    assert var_3 == '"bytes"'


def test_case_0():
    var_0 = '{invalid}'
    var_1 = module_0.tokenize_json(var_0)


def test_case_0():
    var_0 = '   '
    var_1 = module_0.tokenize_json(var_0)


def test_case_0():
    var_0 = '{\n  "key": "value"\n}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.start.line_no
    assert var_4 == 1
    var_5 = var_1.start.column_no
    assert var_5 == 1
    var_6 = var_1.start.char_index
    assert var_6 == 0
    var_7 = var_1.end.line_no
    assert var_7 == 3
    var_8 = var_1.end.column_no
    assert var_8 == 1
    var_9 = len(var_0)
    var_10 = 1
    var_11 = var_9 - var_10
    var_12 = var_1.end.char_index
    var_13 = bool(var_1.end.char_index == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'b'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 2
    var_6 = var_4.string
    assert var_6 == '2'


def test_case_0():
    var_0 = '[10, 20, 30]'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 20
    var_6 = var_4.string
    assert var_6 == '20'


def test_case_0():
    var_0 = '{"x": {"y": 5}}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = 'x'
    var_3 = 'y'
    var_4 = [var_2, var_3]
    var_5 = var_1.lookup_key(var_4)
    var_6 = var_5.value
    assert var_6 == 'y'
    var_7 = var_5.string
    assert var_7 == '"y"'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_32_evaluates_false. Retrieved 10/18 statements.


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



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_scalar_token_null. Retrieved 6/15 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, *args: ([], 0)
    var_3 = lambda self, *args: ('', 0)
    var_4 = 'null'
    var_5 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 7/33 statements.


def test_case_0():
    var_0 = False
    assert var_0 is True
    var_1 = '"key":'
    var_2 = 0
    var_3 = {}
    var_4 = (var_1, var_2)
    var_5 = True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_dict. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_list. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '{}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_scanner_with_empty_string. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_null. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_true. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_false. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_integer. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_float. Retrieved 5/12 statements.
# Partially parsed test_make_scanner_with_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_empty_object. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_empty_array. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_with_nested_structure. Retrieved 5/18 statements.


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

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = '{"key": "value"}'
    var_3 = '{"key": "value"}'
    var_4 = 0



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 7/29 statements.


def test_case_0():
    var_0 = {}
    var_1 = '{"key": "value"}'
    var_2 = '{"key": "value"}'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = 0



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_scanner_parse_object_not_TokenizingJSONObject. Retrieved 26/38 statements.


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = 0
    var_3 = ''
    var_4 = module_0.ScalarToken(var_1, var_2, var_2, var_3)
    var_5 = {var_0: var_4}
    var_6 = 1
    var_7 = (var_5, var_6)
    var_8 = 'Context'
    var_9 = ()
    var_10 = 'parse_array'
    var_11 = 'parse_string'
    var_12 = 'strict'
    var_13 = 'parse_float'
    var_14 = 'parse_int'
    var_15 = 'memo'
    var_16 = module_0.ScalarToken(var_6, var_2, var_2, var_3)
    var_17 = [var_16]
    var_18 = (var_17, var_6)
    var_19 = lambda *args: var_18
    var_20 = 'string'
    var_21 = (var_20, var_6)
    var_22 = lambda *args: var_21
    var_23 = True
    var_24 = {}
    var_25 = '{"key": "value"}'



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = len(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_scalar_token_null. Retrieved 9/17 statements.



def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = lambda self, *args: ([], 0)
    var_3 = lambda self, *args: ('', 0)
    var_4 = 'null'
    var_5 = 0
    var_6 = None
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_4)



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 7/8 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 5/14 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 6/20 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 5/14 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 6/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = None
    var_6 = lambda s, idx: (var_5, idx)

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True

def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True

def test_case_0():
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = 0
    var_4 = (var_0, var_3)
    var_5 = True

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
    var_0 = '{"key": "value" "key2": "value2"}'
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
    var_0 = '{"key":'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True


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
    var_0 = '{key: 123}'
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



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------

# Partially parsed test_tokenize_json_valid_json_returns_token. Retrieved 2/3 statements.
# Partially parsed test_tokenize_json_bytes_input_decoded. Retrieved 2/3 statements.


import typesystem.tokenize.tokenize_json as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True


def test_case_0():
    var_0 = '{"key": "value"'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = b'{"key": "value"}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 15/24 statements.


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = 'value'
    var_5 = 8
    var_6 = 14
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 15
    var_9 = (var_7, var_8)
    var_10 = (var_0, var_1)
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_10, var_5, var_6, var_0)
    var_13 = {var_11: var_12}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 8/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 16/23 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/13 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/10 statements.
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
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)


def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'a'
    var_6 = 2
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 'b'
    var_9 = 8
    var_10 = 9
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = 6
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 13
    var_15 = module_0.ScalarToken(var_6, var_14, var_14, var_0)


def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 10
    var_11 = 16
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 6/18 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 6/15 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 6/14 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 6/14 statements.


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
    var_4 = 'invalid'
    var_5 = 0
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = 3
    var_3 = 'null'
    var_4 = module_0.ScalarToken(var_0, var_1, var_2, var_3)
    var_5 = 'NULL'
    var_6 = module_0.ScalarToken(var_0, var_1, var_2, var_5)
    var_7 = var_4 == var_6
    assert var_7 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_dict_token. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_list_token. Retrieved 5/17 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.
# Partially parsed test_make_scanner_stop_iteration_on_empty_string. Retrieved 5/13 statements.
# Partially parsed test_make_scanner_stop_iteration_on_invalid_char. Retrieved 5/13 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = {}
    var_3 = '"test"'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'true'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'false'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'null'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '42'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '3.14'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('key', idx + 5)
    var_2 = {}
    var_3 = '{"key": "value"}'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = '[1,2]'
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
    var_1 = None
    var_2 = {}
    var_3 = ''
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}
    var_3 = 'invalid'
    var_4 = 0
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_scanner_scalar_string. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_null. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_true. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_false. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_integer. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_scalar_float. Retrieved 5/14 statements.
# Partially parsed test_make_scanner_dict. Retrieved 6/18 statements.
# Partially parsed test_make_scanner_list. Retrieved 6/18 statements.
# Partially parsed test_make_scanner_memo_cleared. Retrieved 5/13 statements.


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
    var_3 = None
    var_4 = '{}'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('', 0)
    var_2 = {}
    var_3 = None
    var_4 = '[]'
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '"test"'
    var_6 = 0



# Parsed testcases at query #5
#--------------------------





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
    var_7 = var_4.string
    assert var_7 == 'null'
    var_8 = var_4.start.line
    assert var_8 == 1
    var_9 = var_4.start.column
    assert var_9 == 1
    var_10 = var_4.start.index
    assert var_10 == 0
    var_11 = var_4.end.line
    assert var_11 == 1
    var_12 = var_4.end.column
    assert var_12 == 4
    var_13 = var_4.end.index
    assert var_13 == 3



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/24 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon_raises. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma_raises. Retrieved 6/14 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote_raises. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_stop_iteration_raises. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True


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
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)


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
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 9
    var_13 = 10
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)


def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 10
    var_11 = 16
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 14/21 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 19/24 statements.
# Partially parsed test_TokenizingJSONObject_whitespace_handling. Retrieved 15/22 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon_raises_error. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma_raises_error. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_quote_on_key_raises_error. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 19/28 statements.


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
    var_12 = {var_7: var_11}
    var_13 = len(var_0)


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
    var_12 = 6
    var_13 = module_0.ScalarToken(var_4, var_12, var_12, var_0)
    var_14 = 2
    var_15 = 13
    var_16 = module_0.ScalarToken(var_14, var_15, var_15, var_0)
    var_17 = {var_7: var_13, var_11: var_16}
    var_18 = len(var_0)


def test_case_0():
    var_0 = '{  "key"  :  "value"  }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 7
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 14
    var_11 = 20
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = len(var_0)

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
    var_0 = '{"key": "value", "key": "another"}'
    var_1 = {}
    var_2 = 0
    var_3 = [var_2]
    var_4 = (var_0, var_2)
    var_5 = True
    var_6 = 'key'
    var_7 = 5
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'value'
    var_10 = 9
    var_11 = 15
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 'another'
    var_14 = 24
    var_15 = 32
    var_16 = module_0.ScalarToken(var_13, var_14, var_15, var_0)
    var_17 = {var_8: var_16}
    var_18 = len(var_0)
    var_19 = 'key'
    var_20 = bool('key' in var_1)
    assert var_20 is True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_parse_object_is_not_TokenizingJSONObject. Retrieved 2/16 statements.


def test_case_0():
    var_0 = ''
    var_1 = 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 21/39 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = lambda s, end, strict: (var_0, end + var_1)
    var_3 = {}
    var_4 = '{"key": "value"}'
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = True
    var_8 = '{"key":'
    var_9 = 7
    var_10 = (var_8, var_9)
    var_11 = True
    var_12 = '{"key": '
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = True
    var_16 = '{"key":  '
    var_17 = 9
    var_18 = (var_16, var_17)
    var_19 = True



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_39_false. Retrieved 10/11 statements.


def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 1
    var_2 = 'obj'
    var_3 = 'end'
    var_4 = lambda s, idx: type(var_0, var_1, {var_2: lambda : idx})()
    var_5 = 1
    var_6 = var_1 + var_5
    var_7 = var_0[var_1:var_6]
    var_8 = ':'
    var_9 = var_7 != var_8
    assert var_9 is False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_61_evaluates_to_false. Retrieved 15/23 statements.



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 0
    var_2 = True
    var_3 = {}
    var_4 = 'value'
    var_5 = 9
    var_6 = 15
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 16
    var_9 = (var_7, var_8)
    var_10 = (var_0, var_1)
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_10, var_5, var_6, var_0)
    var_13 = {var_11: var_12}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_scanner_returns_scalar_token_for_string. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_null. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_true. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_false. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_integer. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_scalar_token_for_float. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_dict_token_for_object. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_returns_list_token_for_array. Retrieved 7/16 statements.
# Partially parsed test_make_scanner_clears_memo_after_scan. Retrieved 7/15 statements.
# Partially parsed test_make_scanner_raises_stop_iteration_on_invalid_input. Retrieved 7/15 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '"test"'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'null'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'true'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'false'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '42'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = '3.14'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = lambda self, string_idx, strict, scan_once, memo, content: ({}, 2)
    var_3 = None
    var_4 = {}
    var_5 = '{}'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = lambda self, string_idx, scan_once: ([], 2)
    var_4 = {}
    var_5 = '[]'
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = lambda self, string, idx, strict: ('test', idx + 6)
    var_2 = None
    var_3 = None
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = '"test"'
    var_8 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = {}
    var_5 = 'invalid'
    var_6 = 0
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #15
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
# Partially parsed test_tokenize_json_nested_structure. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_json_multiline. Retrieved 8/9 statements.


import typesystem.tokenize.tokenize_json as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True

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


def test_case_0():
    var_0 = '{"a": [{"b": true}]}'
    var_1 = module_0.tokenize_json(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [{'b': True}]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '{"a": [{"b": true}]}'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 21
    var_11 = 20
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


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


def test_case_0():
    var_0 = '{"unclosed": '
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 5/8 statements.
# Partially parsed test_TokenizingJSONObject_simple_key_value. Retrieved 12/16 statements.
# Partially parsed test_TokenizingJSONObject_multiple_pairs. Retrieved 17/26 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 6/16 statements.
# Partially parsed test_TokenizingJSONObject_key_not_string. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = '{"key": }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Expecting value'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_39_evaluates_false. Retrieved 15/24 statements.



def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = 1
    var_2 = {}
    var_3 = True
    var_4 = 'value'
    var_5 = 8
    var_6 = 14
    var_7 = module_0.ScalarToken(var_4, var_5, var_6, var_0)
    var_8 = 15
    var_9 = (var_7, var_8)
    var_10 = (var_0, var_1)
    var_11 = 'key'
    var_12 = module_0.ScalarToken(var_10, var_5, var_6, var_0)
    var_13 = {var_11: var_12}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 9/10 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/18 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/23 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/19 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/11 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_memoization. Retrieved 14/20 statements.


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
    var_12 = {var_7: var_11}


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
    var_17 = {var_7: var_13, var_11: var_16}


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
    var_10 = 12
    var_11 = 18
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
    var_0 = '{"key": "value", "key": "value2"}'
    var_1 = {}
    var_2 = 0
    var_3 = [var_2]
    var_4 = (var_0, var_2)
    var_5 = True
    var_6 = 'key'
    var_7 = 5
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'value2'
    var_10 = 23
    var_11 = 29
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}
    var_14 = 'key'
    var_15 = bool('key' in var_1)
    assert var_15 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.tokenize.tokenize_json as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_48_evaluates_to_false. Retrieved 9/23 statements.


def test_case_0():
    var_0 = []
    var_1 = '{"key": "value"}'
    var_2 = 1
    var_3 = {}
    var_4 = (var_1, var_2)
    var_5 = True
    var_6 = len(var_0)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 0



# Parsed testcases at query #21
#--------------------------





def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_json(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/17 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 20/29 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/18 statements.
# Partially parsed test_TokenizingJSONObject_missing_colon. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_comma. Retrieved 5/10 statements.
# Partially parsed test_TokenizingJSONObject_missing_key_quotes. Retrieved 5/9 statements.
# Partially parsed test_TokenizingJSONObject_missing_value. Retrieved 5/9 statements.


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


def test_case_0():
    var_0 = ' { "key" : 123 } '
    var_1 = {}
    var_2 = 1
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 3
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 123
    var_10 = 12
    var_11 = 14
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = {var_8: var_12}

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
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Expecting ',' delimiter"

def test_case_0():
    var_0 = '{key: 123}'
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_TokenizingJSONObject_empty_object. Retrieved 6/9 statements.
# Partially parsed test_TokenizingJSONObject_single_key_value. Retrieved 13/19 statements.
# Partially parsed test_TokenizingJSONObject_multiple_key_values. Retrieved 18/27 statements.
# Partially parsed test_TokenizingJSONObject_with_whitespace. Retrieved 14/20 statements.
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
    var_10 = 14
    var_11 = module_0.ScalarToken(var_8, var_9, var_10, var_0)
    var_12 = {var_7: var_11}


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
    var_10 = module_0.ScalarToken(var_5, var_9, var_9, var_0)
    var_11 = 'b'
    var_12 = 10
    var_13 = 11
    var_14 = module_0.ScalarToken(var_11, var_12, var_13, var_0)
    var_15 = 13
    var_16 = module_0.ScalarToken(var_7, var_15, var_15, var_0)
    var_17 = {var_8: var_10, var_14: var_16}


def test_case_0():
    var_0 = '{ "key" : "value" }'
    var_1 = {}
    var_2 = 0
    var_3 = (var_0, var_2)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 5
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 'value'
    var_10 = 10
    var_11 = 16
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenizing_json_object_empty. Retrieved 6/9 statements.
# Partially parsed test_tokenizing_json_object_single_key. Retrieved 12/15 statements.
# Partially parsed test_tokenizing_json_object_multiple_keys. Retrieved 20/27 statements.
# Partially parsed test_tokenizing_json_object_with_whitespace. Retrieved 13/16 statements.
# Partially parsed test_tokenizing_json_object_missing_colon. Retrieved 6/10 statements.
# Partially parsed test_tokenizing_json_object_missing_comma. Retrieved 6/10 statements.
# Partially parsed test_tokenizing_json_object_invalid_key. Retrieved 6/10 statements.
# Partially parsed test_tokenizing_json_object_stop_iteration_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '{}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True


def test_case_0():
    var_0 = '{"key": 1}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 5
    var_7 = module_0.ScalarToken(var_5, var_4, var_6, var_0)
    var_8 = 8
    var_9 = module_0.ScalarToken(var_4, var_8, var_8, var_0)
    var_10 = {var_7: var_9}


def test_case_0():
    var_0 = '{"a": 1, "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = 0
    var_4 = (var_0, var_1)
    var_5 = True
    var_6 = 'a'
    var_7 = 3
    var_8 = module_0.ScalarToken(var_6, var_5, var_7, var_0)
    var_9 = 'b'
    var_10 = 9
    var_11 = 11
    var_12 = module_0.ScalarToken(var_9, var_10, var_11, var_0)
    var_13 = 6
    var_14 = module_0.ScalarToken(var_5, var_13, var_13, var_0)
    var_15 = 2
    var_16 = 14
    var_17 = module_0.ScalarToken(var_15, var_16, var_16, var_0)
    var_18 = {var_8: var_14, var_12: var_17}


def test_case_0():
    var_0 = '{ "key" : 1 }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = 'key'
    var_6 = 2
    var_7 = 6
    var_8 = module_0.ScalarToken(var_5, var_6, var_7, var_0)
    var_9 = 11
    var_10 = module_0.ScalarToken(var_4, var_9, var_9, var_0)
    var_11 = {var_8: var_10}

def test_case_0():
    var_0 = '{"key" 1}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"a": 1 "b": 2}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{key: 1}'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = '{"key": }'
    var_1 = 0
    var_2 = {}
    var_3 = (var_0, var_1)
    var_4 = True
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------






