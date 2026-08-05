####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_url_opener_requests_success. Retrieved 7/14 statements.
# Partially parsed test_url_opener_requests_error. Retrieved 5/13 statements.
# Partially parsed test_url_opener_urllib_success. Retrieved 5/7 statements.
# Partially parsed test_query_logic_get_with_data. Retrieved 8/10 statements.
# Partially parsed test_query_logic_post_with_data. Retrieved 8/9 statements.
# Partially parsed test_query_logic_append_to_existing_params. Retrieved 8/9 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

def test_case_0():
    var_0 = 'data'
    var_1 = 'method'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'get'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'

def test_case_0():
    var_0 = 'data'
    var_1 = 'method'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'post'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'

def test_case_0():
    var_0 = 'data'
    var_1 = 'method'
    var_2 = 'new'
    var_3 = 'param'
    var_4 = {var_2: var_3}
    var_5 = 'get'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com?old=1'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_query_get_method_with_dict_data. Retrieved 9/11 statements.
# Partially parsed test_query_get_method_with_existing_params. Retrieved 9/11 statements.
# Partially parsed test_query_post_method_with_dict_data. Retrieved 11/13 statements.
# Partially parsed test_query_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_get_with_list_data. Retrieved 13/15 statements.
# Partially parsed test_query_get_method_with_trailing_question_mark. Retrieved 8/10 statements.
# Partially parsed test_query_get_method_with_trailing_ampersand. Retrieved 8/10 statements.


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
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=1&new=2'
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
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=1'
    var_8 = '2'
    var_9 = {var_4: var_8}
    var_10 = {var_2: var_9}
    var_11 = 'http://example.com?a=1&b=2'
    var_12 = 'GET'

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?key=value'

def test_case_0():
    var_0 = 'http://example.com?a=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?a=1&key=value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 9/21 statements.
# Partially parsed test_requests_raises_http_error_on_404. Retrieved 3/14 statements.
# Partially parsed test_requests_with_session_and_encoding. Retrieved 4/16 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)
    assert var_8 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0._requests(var_0, var_1)

def test_case_0():
    var_0 = 'session'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = 'http://example.com'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_predicate_true. Retrieved 10/20 statements.


import requests.api as module_0
import urllib.parse as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get(var_2)
    var_8 = module_1.urlencode(var_7)
    var_9 = 'get'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_predicate_true_with_data_in_kwargs. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #6
#--------------------------




import requests.cookies as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = module_0.MockResponse()
    var_3 = 'http://example.com'
    var_4 = 'method'
    var_5 = 'get'
    var_6 = {var_4: var_5}
    var_7 = module_1._requests(var_3, var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_get_with_existing_params. Retrieved 11/15 statements.
# Partially parsed test_query_with_list_data. Retrieved 9/11 statements.


import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'http://example.com?'
    var_10 = var_9 + var_8
    var_11 = module_1._query(var_0, var_1, var_6)

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'http://example.com?a=b&'
    var_10 = '='

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
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)
    var_11 = module_2._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)

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

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = module_1._query(var_0, var_1, var_6)

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = module_1._query(var_0, var_1, var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_requests_predicate_false_when_method_is_post. Retrieved 9/13 statements.


import requests.api as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'post'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = {var_0: var_2}
    var_7 = 'get'
    var_8 = module_0.get(var_0, var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_requests_predicate_false_when_method_is_post. Retrieved 7/19 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'post'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'success'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_requests_with_session_evaluates_true. Retrieved 8/27 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda url, method, kwargs: (url, var_0)
    var_2 = []
    var_3 = 10
    var_4 = 'session'
    var_5 = 'method'
    var_6 = 'get'
    var_7 = 'http://example.com'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_predicate_false_by_method_not_string. Retrieved 7/9 statements.
# Partially parsed test_query_predicate_false_by_method_not_get. Retrieved 7/9 statements.
# Partially parsed test_query_predicate_false_by_no_data. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_query_get_method_with_existing_query_params. Retrieved 15/20 statements.
# Partially parsed test_query_method_case_insensitivity. Retrieved 9/11 statements.


import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'http://example.com?'
    var_10 = var_9 + var_8
    var_11 = module_1._query(var_0, var_1, var_6)

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = '%'
    var_10 = var_9 in var_8
    var_11 = 'http://example.com?a=b&'
    var_12 = '+'
    var_13 = '%20'
    var_14 = var_11 + var_8

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
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)
    var_9 = 'utf-8'
    var_10 = module_1.encode(var_9)
    var_11 = module_2._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)

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
    var_7 = '0'
    var_8 = '1'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.urlencode(var_11)
    var_13 = 'utf-8'
    var_14 = module_1.encode(var_13)
    var_15 = module_2._query(var_0, var_1, var_6)

import urllib.parse as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'gEt'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_0.urlencode(var_7)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 9/20 statements.
# Partially parsed test_requests_post_with_data_failure. Retrieved 9/18 statements.
# Partially parsed test_requests_session_usage. Retrieved 4/12 statements.
# Partially parsed test_requests_encoding_application. Retrieved 7/12 statements.


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
    var_8 = module_0._requests(var_7, var_6)
    assert var_8 == '<html>content</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'post'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'latin-1'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'encoded_text'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_query_predicate_true. Retrieved 12/14 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=true'
    var_8 = 'GET'
    var_9 = 'new'
    var_10 = 'param'
    var_11 = {var_9: var_10}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_url_opener_requests_get_success. Retrieved 7/17 statements.
# Failed to parse test_url_opener_requests_error_raises_exception.
# Partially parsed test_url_opener_urllib_logic_query_params. Retrieved 8/16 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == '<html>success</html>'

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_requests_with_session_evaluates_true. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_requests_status_code_not_in_success_range. Retrieved 9/33 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = 10
    var_5 = 'session'
    var_6 = 'method'
    var_7 = 'get'
    var_8 = 'http://example.com'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_url_opener_calls_requests_when_available. Retrieved 7/14 statements.
# Partially parsed test_url_opener_calls_urllib_when_requests_not_available. Retrieved 7/11 statements.
# Partially parsed test_url_opener_raises_http_error_on_bad_status. Retrieved 5/15 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = b'html_content'
    var_1 = lambda : var_0
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 15/31 statements.
# Partially parsed test_requests_encoding_application. Retrieved 7/22 statements.
# Partially parsed test_requests_raises_http_error. Retrieved 5/23 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'headers'
    var_3 = 'auth'
    var_4 = 'method'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'val'
    var_8 = {var_6: var_7}
    var_9 = 10
    var_10 = {var_4: var_5, var_0: var_8, var_1: var_9}
    var_11 = 'http://example.com'
    var_12 = module_0._requests(var_11, var_10)
    assert var_12 == '<html>success</html>'
    var_13 = 'http://example.com?key=val'
    var_14 = {var_6: var_7}

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'content'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0._requests(var_3, var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 13/24 statements.
# Partially parsed test_requests_failure_raises_http_error. Retrieved 4/16 statements.
# Partially parsed test_requests_with_session. Retrieved 6/17 statements.
# Partially parsed test_requests_encoding_application. Retrieved 7/16 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = 'a'
    var_5 = '1'
    var_6 = {var_4: var_5}
    var_7 = 10
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0._requests(var_9, var_8)
    assert var_10 == '<html>content</html>'
    var_11 = 'http://example.com?a=1'
    var_12 = {var_4: var_5}

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0._requests(var_1, var_2)

def test_case_0():
    var_0 = 'timeout'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'session'
    var_4 = 2
    var_5 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'encoding'
    var_3 = 'utf-8'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'data'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 3/20 statements.


import requests.cookies as module_0

def test_case_0():
    var_0 = 404
    var_1 = 'http://example.com'
    var_2 = module_0.MockResponse(var_0)

import requests.cookies as module_0

def test_case_0():
    var_0 = 404
    var_1 = module_0.MockResponse(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 11/20 statements.
# Partially parsed test_requests_raises_http_error_on_404. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'session'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = 'http://example.com'
    var_10 = 'http://example.com?a=b'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com/bad'
    var_4 = module_0._requests(var_3, var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_list_data_get. Retrieved 9/13 statements.


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

import pyquery.openers as module_0

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
    var_9 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None
    var_5 = module_0._query(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'list_format'
    var_8 = 'http://example.com?a=%5Ba%2C+b%5D'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?k=v'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 10/25 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 9/22 statements.
# Partially parsed test_requests_http_error_raises. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'headers'
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 'method'
    var_6 = 'session'
    var_7 = 'get'
    var_8 = 10
    var_9 = 'http://example.com'

def test_case_0():
    var_0 = 'params'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 'method'
    var_4 = 'session'
    var_5 = 'encoding'
    var_6 = 'get'
    var_7 = 'utf-8'
    var_8 = 'http://example.com'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'http://example.com'
    var_6 = 'HTTPError was not raised'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_predicate_false_by_method_type. Retrieved 7/21 statements.
# Partially parsed test_query_predicate_false_by_method_value. Retrieved 7/21 statements.
# Partially parsed test_query_predicate_false_by_no_data. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_requests_success_get_with_params. Retrieved 10/19 statements.
# Partially parsed test_requests_http_error_raises. Retrieved 4/14 statements.
# Partially parsed test_requests_encoding_assignment. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'params'
    var_2 = 'session'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'a'
    var_6 = '1'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 'http://example.com'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 9/32 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 8/26 statements.
# Partially parsed test_requests_raises_http_error_on_failure. Retrieved 5/25 statements.
# Partially parsed test_requests_with_session. Retrieved 4/22 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'params'
    var_2 = 'http://example.com'
    var_3 = 'method'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = module_0._requests(var_2, var_7)
    assert var_8 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'encoding'
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0._requests(var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_predicate_false_due_to_method_not_string. Retrieved 8/23 statements.
# Partially parsed test_query_predicate_false_due_to_no_data. Retrieved 3/17 statements.
# Partially parsed test_query_predicate_false_due_to_wrong_method. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 123

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_requests_fails_on_non_2xx_status_code. Retrieved 9/32 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'requests'
    var_1 = 'timeout'
    var_2 = 'params'
    var_3 = 'method'
    var_4 = 'get'
    var_5 = 10
    var_6 = {var_3: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0._requests(var_7, var_6)



# Parsed testcases at query #13
#--------------------------




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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=1&new=2'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)

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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = 'http://example.com'
    var_4 = None
    var_5 = module_0._query(var_0, var_1, var_2)

import urllib.parse as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = [var_3, var_4]
    var_8 = False
    var_9 = module_0.urlencode(var_7, var_8)
    var_10 = 'http://example.com?'
    var_11 = [var_3, var_4]
    var_12 = module_0.urlencode(var_11)
    var_13 = var_10 + var_12
    var_14 = None
    var_15 = module_1._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?existing=1&new=2'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'http://example.com?new=2'
    var_8 = None
    var_9 = module_0._query(var_0, var_1, var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_requests_raises_http_error_on_non_success_status. Retrieved 11/28 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'params'
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = 'http://example.com'
    var_5 = 'method'
    var_6 = 'timeout'
    var_7 = 'get'
    var_8 = 5
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0._requests(var_4, var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_requests_predicate_false_on_post. Retrieved 7/18 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'POST'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'success'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_urllib_get_request_with_data. Retrieved 12/21 statements.
# Partially parsed test_urllib_post_request_with_data. Retrieved 16/25 statements.
# Partially parsed test_urllib_with_timeout. Retrieved 10/19 statements.


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
    var_11 = 60

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
    var_13 = '__main__'
    var_14 = module_2._urllib(var_0, var_7)
    var_15 = 60

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'GET'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'http://example.com'
    var_7 = None
    var_8 = module_0._urllib(var_0, var_5)
    var_9 = 10



