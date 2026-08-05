####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_get_with_dict_data. Retrieved 9/11 statements.
# Partially parsed test_query_get_with_existing_query_params. Retrieved 9/11 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 11/13 statements.
# Partially parsed test_query_no_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_trailing_ampersand. Retrieved 9/11 statements.
# Partially parsed test_query_get_with_trailing_question_mark. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None

import email._encoded_words as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = 'key=value'
    var_9 = 'utf-8'
    var_10 = module_0.encode(var_9)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None

def test_case_0():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?c=d'
    var_8 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 11/28 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 10/26 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 8/30 statements.
# Partially parsed test_requests_with_session. Retrieved 6/25 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'params'
    var_2 = 'HTTPError'
    var_3 = {}
    var_4 = 'method'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_4: var_5, var_0: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == '<html>success</html>'
    var_10 = {}

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'HTTPError'
    var_2 = {}
    var_3 = 'method'
    var_4 = 'encoding'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == '<html>utf8</html>'

import pyquery.openers as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = 'http://example.com'
    var_5 = module_0._requests(var_4, var_3)
    var_6 = 'HTTPError was not raised'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.AssertionError(*var_7, **var_8)

def test_case_0():
    var_0 = 'HTTPError'
    var_1 = {}
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_case_insensitive_get. Retrieved 13/15 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=val'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)
    var_12 = module_2._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_7, var_11))
    assert var_13 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None
    var_5 = module_0._query(var_0, var_1, var_2)
    var_6 = bool(var_5 == (var_3, var_4))
    assert var_6 is True

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = [var_3, var_4]
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)
    var_12 = module_2._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_7, var_11))
    assert var_13 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'gEt'
    var_2 = 'data'
    var_3 = 'id'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example/example.com?id=1'
    var_8 = 'example.com'
    var_9 = 'http://example.com?id=1'
    var_10 = 'http://example.com?id=1'
    var_11 = None
    var_12 = module_0._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_10, var_11))
    assert var_13 is True

import email._encoded_words as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = module_1._query(var_0, var_1, var_4)
    var_9 = bool(var_8 == (var_5, var_7))
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_url_opener_requests_get_success. Retrieved 6/17 statements.
# Partially parsed test_url_opener_requests_get_with_data. Retrieved 13/19 statements.
# Partially parsed test_url_opener_requests_error. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = 'http://example.com'

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com?'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = var_7 + var_9
    var_11 = 'http://example.com'
    var_12 = module_1.url_opener(var_11, var_6)
    assert var_12 == '<html>data</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_urllib_get_request_with_data_appends_to_query_string. Retrieved 11/16 statements.
# Partially parsed test_urllib_post_request_keeps_data_in_body. Retrieved 14/19 statements.
# Partially parsed test_urllib_get_request_with_existing_params_appends_ampersand. Retrieved 11/16 statements.
# Partially parsed test_urllib_with_timeout_parameter. Retrieved 10/14 statements.
# Partially parsed test_urllib_handles_list_data_in_get. Retrieved 14/19 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?key=value'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'POST'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com'
    var_9 = {var_4: var_5}
    var_10 = module_0.urlencode(var_9)
    var_11 = 'utf-8'
    var_12 = module_1.encode(var_11)
    var_13 = module_2._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'param'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?existing=true&new=param'
    var_9 = None
    var_10 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = None
    var_8 = module_0._urllib(var_0, var_5)
    var_9 = 30

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'http://example.com?'
    var_9 = [var_4, var_5]
    var_10 = module_0.urlencode(var_9)
    var_11 = var_8 + var_10
    var_12 = None
    var_13 = module_1._urllib(var_0, var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_requests_with_session_evaluates_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_get_with_dict_data. Retrieved 9/11 statements.
# Partially parsed test_query_get_with_existing_params. Retrieved 9/11 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 12/14 statements.
# Partially parsed test_query_no_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_list_data. Retrieved 11/13 statements.
# Partially parsed test_query_get_with_trailing_ampersand. Retrieved 8/10 statements.
# Partially parsed test_query_case_insensitive_method. Retrieved 11/13 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None

import urllib.parse as module_0
import email._encoded_words as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_3, var_4]
    var_8 = module_0.urlencode(var_7)
    var_9 = 'http://example.com?'
    var_10 = var_9 + var_8

def test_case_0():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'gEt'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?'
    var_8 = {var_3: var_4}
    var_9 = module_0.urlencode(var_8)
    var_10 = var_7 + var_9



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_not_string. Retrieved 5/10 statements.
# Partially parsed test_query_predicate_false_due_to_method_not_get. Retrieved 5/10 statements.
# Partially parsed test_query_predicate_false_due_to_no_data. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'name=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'name=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 8/27 statements.
# Partially parsed test_requests_status_code_range_is_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = 'session'
    var_3 = 'method'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 5
    var_7 = 'http://example.com'

def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://test.com'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 14/27 statements.
# Partially parsed test_requests_post_with_encoding. Retrieved 14/26 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 8/22 statements.
# Partially parsed test_requests_uses_session. Retrieved 6/20 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'headers'
    var_2 = 'timeout'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'get'
    var_7 = 'key'
    var_8 = 'val'
    var_9 = {var_7: var_8}
    var_10 = 10
    var_11 = {var_5: var_6, var_0: var_9, var_2: var_10}
    var_12 = 'http://example.com'
    var_13 = module_0._requests(var_12, var_11)
    assert var_13 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 'headers'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'method'
    var_5 = 'encoding'
    var_6 = 'post'
    var_7 = 'id'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = 'utf-8'
    var_11 = {var_4: var_6, var_0: var_9, var_5: var_10}
    var_12 = 'http://example.com/post'
    var_13 = module_0._requests(var_12, var_11)
    assert var_13 == 'Created'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'get'
    var_5 = {var_3: var_4}
    var_6 = 'http://example.com/fail'
    var_7 = module_0._requests(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_url_opener_predicate_false_when_has_requests_is_false. Retrieved 3/11 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 11/18 statements.
# Partially parsed test_requests_raises_http_error_on_404. Retrieved 8/17 statements.
# Partially parsed test_requests_with_session_and_encoding. Retrieved 10/22 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = 10
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == '<html>content</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'get'
    var_5 = {var_3: var_4}
    var_6 = 'http://example.com/bad'
    var_7 = module_0._requests(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'HTTPError'

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'session'
    var_5 = 'encoding'
    var_6 = 'get'
    var_7 = 'utf-8'
    var_8 = 2
    var_9 = 'http://example.com'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_type. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 11/35 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests'
    var_1 = 'http://example.com'
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = []
    var_5 = 10
    var_6 = 'http://example.com'
    var_7 = 'method'
    var_8 = 'get'
    var_9 = {var_7: var_8}
    var_10 = module_0._requests(var_6, var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_url_opener_predicate_is_false_when_has_request_is_none. Retrieved 3/9 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_url_opener_predicate_is_false_when_has_request_is_false. Retrieved 3/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_requests_success_get. Retrieved 13/29 statements.
# Partially parsed test_requests_failure_raises_http_error. Retrieved 11/27 statements.
# Partially parsed test_requests_with_session_and_encoding. Retrieved 6/23 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = 10
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == '<html>success</html>'
    var_11 = 'http://example.com?key=val'
    var_12 = {var_4: var_5}

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'Content-Type'
    var_1 = 'text/plain'
    var_2 = 'params'
    var_3 = [var_2]
    var_4 = 5
    var_5 = 'id'
    var_6 = '123'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'session'
    var_3 = 'encoding'
    var_4 = 'utf-8'
    var_5 = 'http://example.com'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status_code. Retrieved 7/19 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = 'http://example.com'
    var_3 = 'method'
    var_4 = 'get'
    var_5 = {var_3: var_4}
    var_6 = module_0._requests(var_2, var_5)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 11/25 statements.
# Partially parsed test_requests_with_encoding. Retrieved 10/23 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 8/21 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'headers'
    var_2 = 'timeout'
    var_3 = [var_0, var_1, var_2]
    var_4 = 10
    var_5 = 'method'
    var_6 = 'get'
    var_7 = 5
    var_8 = {var_5: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 10
    var_3 = 'method'
    var_4 = 'encoding'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == 'content'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 10
    var_3 = 'method'
    var_4 = 'get'
    var_5 = {var_3: var_4}
    var_6 = 'http://example.com'
    var_7 = module_0._requests(var_6, var_5)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=b&c=d'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import urllib.parse as module_0
import email._encoded_words as module_1
import pyquery.openers as module_2

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = [var_3, var_4]
    var_9 = module_0.urlencode(var_8)
    var_10 = 'utf-8'
    var_11 = module_1.encode(var_10)
    var_12 = module_2._query(var_0, var_1, var_6)
    var_13 = bool(var_12 == (var_7, var_11))
    assert var_13 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None
    var_5 = module_0._query(var_0, var_1, var_2)
    var_6 = bool(var_5 == (var_3, var_4))
    assert var_6 is True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=val'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

import email._encoded_words as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = 'utf-8'
    var_7 = module_0.encode(var_6)
    var_8 = module_1._query(var_0, var_1, var_4)
    var_9 = bool(var_8 == (var_5, var_7))
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 14/28 statements.
# Partially parsed test_requests_error_raises_http_error. Retrieved 8/22 statements.
# Partially parsed test_requests_with_session_and_encoding. Retrieved 10/27 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'headers'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = 'http://example.com'
    var_11 = module_0._requests(var_10, var_9)
    assert var_11 == '<html>content</html>'
    var_12 = 'http://example.com?a=b'
    var_13 = {var_5: var_6}

import pyquery.openers as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = {}
    var_4 = 'http://example.com'
    var_5 = module_0._requests(var_4, var_3)
    var_6 = 'HTTPError was not raised'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.AssertionError(*var_7, **var_8)

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'session'
    var_5 = 'encoding'
    var_6 = 'post'
    var_7 = 'utf-8'
    var_8 = 2
    var_9 = 'http://example.com'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_requests_predicate_true. Retrieved 8/26 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 'other_arg'
    var_3 = 5
    var_4 = 'ignored'
    var_5 = {var_0: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = module_0._requests(var_6, var_5)
    assert var_7 == 'success'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 7/20 statements.
# Partially parsed test_requests_get_with_session. Retrieved 4/18 statements.
# Partially parsed test_requests_encoding_application. Retrieved 7/16 statements.
# Partially parsed test_requests_http_error_raises. Retrieved 5/14 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == '<html>success</html>'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'latin-1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'utf-8 content'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 12/28 statements.
# Partially parsed test_requests_get_with_data_encoding. Retrieved 16/34 statements.
# Partially parsed test_requests_http_error. Retrieved 9/29 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'requests'
    var_5 = 'HTTPError'
    var_6 = 'http://example.com'
    var_7 = 'method'
    var_8 = 'get'
    var_9 = 10
    var_10 = {var_7: var_8, var_1: var_9}
    var_11 = module_0._requests(var_6, var_10)
    assert var_11 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'requests'
    var_4 = 'HTTPError'
    var_5 = 'http://example.com'
    var_6 = 'method'
    var_7 = 'data'
    var_8 = 'encoding'
    var_9 = 'get'
    var_10 = 'key'
    var_11 = 'val'
    var_12 = {var_10: var_11}
    var_13 = 'utf-8'
    var_14 = {var_6: var_9, var_7: var_12, var_8: var_13}
    var_15 = module_0._requests(var_5, var_14)
    assert var_15 == 'encoded'

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'requests'
    var_3 = 'HTTPError'
    var_4 = 'http://example.com'
    var_5 = 'method'
    var_6 = 'get'
    var_7 = {var_5: var_6}
    var_8 = module_0._requests(var_4, var_7)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_predicate_true. Retrieved 7/28 statements.


def test_case_0():
    var_0 = 'http://example.com/api'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'key=value'
    var_8 = '?'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 10/26 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 10/21 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 7/20 statements.
# Partially parsed test_requests_uses_session. Retrieved 7/22 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'method'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_4: var_5, var_1: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'encoding'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'http://example.com'
    var_9 = module_0._requests(var_8, var_7)
    assert var_9 == '<html>utf8</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'
    var_6 = 5



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_url_opener_calls_requests_with_correct_params. Retrieved 9/20 statements.
# Partially parsed test_url_opener_raises_http_error_on_failure. Retrieved 5/15 statements.
# Partially parsed test_url_opener_calls_urllib_when_requests_not_available. Retrieved 5/9 statements.
# Partially parsed test_query_logic_appends_data_to_get_url. Retrieved 9/17 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 5
    var_5 = 'utf-8'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    var_9 = 'key=value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_url_opener_with_requests. Retrieved 8/14 statements.
# Partially parsed test_url_opener_requests_error. Retrieved 5/13 statements.
# Partially parsed test_url_opener_with_urllib. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_query_params. Retrieved 14/20 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == 'success'
    var_7 = None

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = b'content'
    var_1 = lambda : var_0
    var_2 = 'http://test.com'
    var_3 = 'method'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 5
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0.url_opener(var_2, var_7)

import pyquery.openers as module_0
import urllib.parse as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'http://test.com'
    var_4 = 'method'
    var_5 = 'data'
    var_6 = 'get'
    var_7 = {var_4: var_6, var_5: var_2}
    var_8 = module_0.url_opener(var_3, var_7)
    var_9 = 'http://test.com?'
    var_10 = module_1.urlencode(var_2)
    var_11 = var_9 + var_10
    var_12 = 'url'
    var_13 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 11/33 statements.
# Partially parsed test_requests_error_raises_exception. Retrieved 9/31 statements.
# Partially parsed test_requests_with_session. Retrieved 8/30 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests'
    var_1 = 'timeout'
    var_2 = 'params'
    var_3 = 'HTTPError'
    var_4 = {}
    var_5 = 'method'
    var_6 = 'get'
    var_7 = 10
    var_8 = {var_5: var_6, var_1: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == '<html></html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests'
    var_1 = 'timeout'
    var_2 = 'HTTPError'
    var_3 = {}
    var_4 = 'method'
    var_5 = 'get'
    var_6 = {var_4: var_5}
    var_7 = 'http://example.com/error'
    var_8 = module_0._requests(var_7, var_6)

def test_case_0():
    var_0 = 'requests'
    var_1 = 'timeout'
    var_2 = 'HTTPError'
    var_3 = {}
    var_4 = 'method'
    var_5 = 'session'
    var_6 = 'get'
    var_7 = 'http://example.com'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_predicate_false_method_not_string. Retrieved 12/25 statements.
# Partially parsed test_query_predicate_false_method_not_get. Retrieved 8/15 statements.
# Partially parsed test_query_predicate_false_no_data. Retrieved 4/10 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'urlencode'
    var_8 = {}
    var_9 = module_0.get(var_7, **var_8)
    var_10 = 'urlencode'
    var_11 = 123
    var_12 = 'urlencode'

def test_case_0():
    var_0 = 'urlencode'
    var_1 = 'http://example.com'
    var_2 = 'POST'
    var_3 = 'data'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'urlencode'
    var_1 = 'http://example.com'
    var_2 = 'get'
    var_3 = {}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 16/36 statements.
# Partially parsed test_requests_failure_raises_error. Retrieved 9/26 statements.
# Partially parsed test_requests_uses_session. Retrieved 6/25 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'headers'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'encoding'
    var_7 = 'get'
    var_8 = 'key'
    var_9 = 'val'
    var_10 = {var_8: var_9}
    var_11 = 'utf-8'
    var_12 = {var_5: var_7, var_0: var_10, var_6: var_11}
    var_13 = 'urllib.parse'
    var_14 = 'http://example.com'
    var_15 = module_0._requests(var_14, var_12)
    assert var_15 == '<html>content</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'get'
    var_5 = 10
    var_6 = {var_3: var_4, var_0: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'POST'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'param'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_url_opener_get_with_data_params. Retrieved 8/10 statements.
# Partially parsed test_url_opener_get_with_existing_query_params. Retrieved 8/10 statements.
# Partially parsed test_url_opener_post_with_data. Retrieved 7/8 statements.
# Partially parsed test_url_opener_query_logic_no_data. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'http://example.com?key=value'
    var_7 = 'get'

def test_case_0():
    var_0 = 'data'
    var_1 = 'a'
    var_2 = '1'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com?existing=true'
    var_6 = 'http://example.com?existing=true&a=1'
    var_7 = 'get'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'post'

def test_case_0():
    var_0 = {}
    var_1 = 'http://example.com'
    var_2 = 'get'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 11/20 statements.
# Partially parsed test_requests_raises_http_error_on_failure. Retrieved 5/13 statements.
# Partially parsed test_requests_with_encoding_and_params. Retrieved 6/14 statements.
# Partially parsed test_requests_uses_default_timeout. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'session'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'a'
    var_6 = '1'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = 'http://example.com'
    var_10 = 'http://example.com?a=1'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0._requests(var_3, var_2)

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'session'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'timeout'
    var_2 = 5
    var_3 = 'http://example.com'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status_code. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_url_opener_predicate_false_when_has_request_is_false. Retrieved 10/20 statements.


import requests.api as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = globals()
    var_1 = 'HAS_REQUEST'
    var_2 = None
    var_3 = {}
    var_4 = module_0.get(var_1, var_2, **var_3)
    var_5 = False
    var_6 = 'http://example.com'
    var_7 = {}
    var_8 = module_1.url_opener(var_6, var_7)
    var_9 = 'HAS_REQUEST'
    var_10 = 'HAS_REQUEST'



