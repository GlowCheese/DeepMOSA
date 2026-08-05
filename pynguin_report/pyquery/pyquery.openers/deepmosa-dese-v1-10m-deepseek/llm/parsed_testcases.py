####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_existing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_trailing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_trailing_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_dict_data_and_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_string_data_and_post_method. Retrieved 5/6 statements.
# Partially parsed test_query_without_data. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com?existing=1'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com?'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com?existing=1&'
    var_6 = 'GET'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'POST'

def test_case_0():
    var_0 = 'data'
    var_1 = 'raw_string'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 'POST'

def test_case_0():
    var_0 = {}
    var_1 = 'http://example.com'
    var_2 = 'GET'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_requests_get_success. Retrieved 9/10 statements.
# Partially parsed test_requests_get_with_data. Retrieved 13/14 statements.
# Partially parsed test_requests_post_success. Retrieved 9/10 statements.
# Partially parsed test_requests_with_session. Retrieved 9/11 statements.
# Partially parsed test_requests_with_timeout. Retrieved 11/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0._requests(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'post'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_2: var_5, var_3: var_6, var_4: var_0}
    var_8 = module_1._requests(var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpstat.us/404'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = 5
    var_8 = None
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = module_0._requests(var_0, var_9)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_requests_get_without_session_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_requests_get_with_query_params. Retrieved 11/12 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 9/10 statements.
# Partially parsed test_requests_post_without_session. Retrieved 11/12 statements.
# Partially parsed test_requests_with_session_get. Retrieved 9/11 statements.
# Partially parsed test_requests_with_session_post. Retrieved 13/15 statements.
# Partially parsed test_requests_with_allowed_args. Retrieved 11/12 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_2: var_5, var_3: var_0, var_4: var_6}
    var_8 = module_1._requests(var_1, var_7)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'data'
    var_5 = 'timeout'
    var_6 = 'post'
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 10
    var_11 = {var_2: var_6, var_3: var_0, var_4: var_9, var_5: var_10}
    var_12 = module_1._requests(var_1, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'headers'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'User-Agent'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_url_opener_with_requests_and_get_method. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_urllib_and_post_method. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_requests_and_session. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_requests_and_encoding. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'timeout'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 10
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = module_0.Session()
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_1.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 'gbk'
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://httpbin.org/status/404'
    var_6 = module_0.url_opener(var_5, var_4)



# Parsed testcases at query #5
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)



# Parsed testcases at query #6
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_status_code_within_success_range. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 'url'
    var_4 = 'reason'
    var_5 = 'headers'
    var_6 = 200
    var_7 = ''
    var_8 = {}
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_7, var_5: var_8}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_10_evaluates_to_false. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #9
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_2: var_5, var_3: var_0, var_4: var_6}
    var_8 = module_1._requests(var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_urllib_get_with_data. Retrieved 8/10 statements.
# Partially parsed test_urllib_post_with_data. Retrieved 8/10 statements.
# Partially parsed test_urllib_get_without_data. Retrieved 4/6 statements.
# Partially parsed test_urllib_with_timeout. Retrieved 6/8 statements.
# Partially parsed test_urllib_default_timeout. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'post'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_with_dict_data_and_get_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_list_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_tuple_data_and_get_method. Retrieved 8/9 statements.
# Partially parsed test_query_with_dict_data_and_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_non_dict_list_tuple_data. Retrieved 5/6 statements.
# Partially parsed test_query_with_get_method_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_with_get_method_and_url_has_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_url_ends_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_with_get_method_and_url_ends_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_empty_data_dict_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_empty_list_data_and_get_method. Retrieved 5/6 statements.
# Partially parsed test_query_with_empty_tuple_data_and_get_method. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = {var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=param&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = []
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = ()
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_requests_get_with_session. Retrieved 4/18 statements.
# Partially parsed test_requests_post_no_session. Retrieved 7/21 statements.
# Partially parsed test_requests_get_with_data_in_url. Retrieved 9/23 statements.
# Partially parsed test_requests_http_error. Retrieved 5/20 statements.


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
    var_3 = 'post'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == 'post response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)
    assert var_8 == 'data in url'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_url_already_ends_with_ampersand. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'http://example.com/api?param1=value1&'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param2=value2'
    var_4 = {var_2: var_3}
    var_5 = '?'
    var_6 = '&'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_line12_false. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'param=1'
    var_4 = {var_2: var_3}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_url_opener_uses_requests_when_available. Retrieved 6/11 statements.
# Partially parsed test_url_opener_uses_urllib_when_requests_not_available. Retrieved 6/11 statements.
# Partially parsed test_url_opener_with_urllib_and_data_in_get. Retrieved 10/15 statements.
# Partially parsed test_url_opener_with_requests_and_encoding. Retrieved 8/13 statements.
# Partially parsed test_url_opener_with_requests_session. Retrieved 8/14 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = True
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'get'
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)

import requests.api as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'data'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = module_0.get(var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = True
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.url_opener(var_1, var_6)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Session()
    var_2 = 'http://example.com'
    var_3 = 'method'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = {var_3: var_5, var_4: var_1}
    var_7 = module_1.url_opener(var_2, var_6)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_url_opener_with_requests_get_and_data. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_requests_session. Retrieved 9/11 statements.
# Partially parsed test_url_opener_with_requests_encoding. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0.url_opener(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'timeout'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 5
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0.url_opener(var_0, var_9)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 5
    var_7 = {var_2: var_5, var_3: var_0, var_4: var_6}
    var_8 = module_1.url_opener(var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 5
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_query_get_with_dict_data_and_no_query_string. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_dict_data_and_existing_query_string. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_dict_data_and_trailing_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_dict_data_and_trailing_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_tuple_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_tuple_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_post_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_query_get_with_uppercase_method. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_lowercase_method. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'already_encoded'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = '1'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_urllib_get_with_data_appends_to_url. Retrieved 10/11 statements.


import requests.api as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.get(var_1)
    var_9 = module_1._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._urllib(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._urllib(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._urllib(var_0, var_3)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.url_opener(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'post'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.url_opener(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?existing=param'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.url_opener(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.url_opener(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com?param=1&'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = None
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = module_0.url_opener(var_0, var_11)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

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
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_requests_get_without_data. Retrieved 7/8 statements.
# Partially parsed test_requests_get_with_data. Retrieved 11/12 statements.
# Partially parsed test_requests_post. Retrieved 11/12 statements.
# Partially parsed test_requests_with_session. Retrieved 9/10 statements.
# Partially parsed test_requests_with_timeout. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'utf-8'
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'post'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'utf-8'
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0._requests(var_0, var_9)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = module_0.Session()
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'encoding'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_2: var_5, var_3: var_1, var_4: var_6}
    var_8 = module_1._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 10
    var_6 = 'utf-8'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_get_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_tuple_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_ending_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_ending_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_empty_data. Retrieved 5/6 statements.
# Partially parsed test_query_post_with_dict_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_list_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_with_tuple_data. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_lowercase_get. Retrieved 7/8 statements.
# Partially parsed test_query_get_with_none_data. Retrieved 5/6 statements.
# Partially parsed test_query_post_with_none_data. Retrieved 5/6 statements.
# Partially parsed test_query_post_with_string_data. Retrieved 5/6 statements.
# Partially parsed test_query_get_with_string_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_requests_get_with_data. Retrieved 9/10 statements.
# Partially parsed test_requests_post_no_data. Retrieved 5/6 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/8 statements.
# Partially parsed test_requests_with_session. Retrieved 7/9 statements.
# Partially parsed test_requests_with_timeout. Retrieved 7/8 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'post'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'get'
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = module_1._requests(var_1, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpbin.org/status/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._requests(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #5
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}
    var_3 = None
    var_4 = module_0._query(var_0, var_1, var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_requests_get_no_session_no_data. Retrieved 7/9 statements.
# Partially parsed test_requests_get_with_session. Retrieved 9/11 statements.
# Partially parsed test_requests_get_with_encoding. Retrieved 9/10 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)

import requests.sessions as module_0
import pyquery.openers as module_1

def test_case_0():
    var_0 = module_0.Session()
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'session'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 10
    var_7 = {var_2: var_5, var_3: var_0, var_4: var_6}
    var_8 = module_1._requests(var_1, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'utf-8'
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0._requests(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://httpstat.us/404'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_requests_get_with_data_and_no_session. Retrieved 25/30 statements.
# Partially parsed test_requests_get_with_query_data. Retrieved 23/28 statements.
# Partially parsed test_requests_get_with_session. Retrieved 20/27 statements.
# Partially parsed test_requests_post_with_data. Retrieved 21/26 statements.
# Partially parsed test_requests_http_error. Retrieved 21/26 statements.
# Partially parsed test_requests_with_timeout. Retrieved 21/26 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'utf-8'
    var_10 = 10
    var_11 = {var_1: var_5, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = 'Response'
    var_13 = ()
    var_14 = 'status_code'
    var_15 = 'text'
    var_16 = 'url'
    var_17 = 'reason'
    var_18 = 'headers'
    var_19 = 200
    var_20 = 'success'
    var_21 = 'OK'
    var_22 = {}
    var_23 = {var_14: var_19, var_15: var_20, var_3: var_9, var_16: var_0, var_17: var_21, var_18: var_22}
    var_24 = module_0._requests(var_0, var_11)
    assert var_24 == 'success'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'utf-8'
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = 'Response'
    var_11 = ()
    var_12 = 'status_code'
    var_13 = 'text'
    var_14 = 'url'
    var_15 = 'reason'
    var_16 = 'headers'
    var_17 = 200
    var_18 = 'success'
    var_19 = 'OK'
    var_20 = {}
    var_21 = {var_12: var_17, var_13: var_18, var_3: var_8, var_14: var_0, var_15: var_19, var_16: var_20}
    var_22 = module_0._requests(var_0, var_9)
    assert var_22 == 'success'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'Session'
    var_2 = ()
    var_3 = 'get'
    var_4 = 'Response'
    var_5 = ()
    var_6 = 'status_code'
    var_7 = 'text'
    var_8 = 'encoding'
    var_9 = 'url'
    var_10 = 'reason'
    var_11 = 'headers'
    var_12 = 200
    var_13 = 'session_ok'
    var_14 = 'utf-8'
    var_15 = 'OK'
    var_16 = {}
    var_17 = {var_6: var_12, var_7: var_13, var_8: var_14, var_9: var_0, var_10: var_15, var_11: var_16}
    var_18 = 'method'
    var_19 = 'session'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'encoding'
    var_4 = 'post'
    var_5 = 'data_string'
    var_6 = 'utf-8'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Response'
    var_9 = ()
    var_10 = 'status_code'
    var_11 = 'text'
    var_12 = 'url'
    var_13 = 'reason'
    var_14 = 'headers'
    var_15 = 200
    var_16 = 'posted'
    var_17 = 'OK'
    var_18 = {}
    var_19 = {var_10: var_15, var_11: var_16, var_3: var_6, var_12: var_0, var_13: var_17, var_14: var_18}
    var_20 = module_0._requests(var_0, var_7)
    assert var_20 == 'posted'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Response'
    var_7 = ()
    var_8 = 'status_code'
    var_9 = 'text'
    var_10 = 'url'
    var_11 = 'reason'
    var_12 = 'headers'
    var_13 = 404
    var_14 = 'not found'
    var_15 = 'Not Found'
    var_16 = {}
    var_17 = {var_8: var_13, var_9: var_14, var_2: var_4, var_10: var_0, var_11: var_15, var_12: var_16}
    var_18 = False
    var_19 = module_0._requests(var_0, var_5)
    var_20 = True

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'encoding'
    var_4 = 'get'
    var_5 = 5
    var_6 = 'utf-8'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'Response'
    var_9 = ()
    var_10 = 'status_code'
    var_11 = 'text'
    var_12 = 'url'
    var_13 = 'reason'
    var_14 = 'headers'
    var_15 = 200
    var_16 = 'timeout_test'
    var_17 = 'OK'
    var_18 = {}
    var_19 = {var_10: var_15, var_11: var_16, var_3: var_6, var_12: var_0, var_13: var_17, var_14: var_18}
    var_20 = module_0._requests(var_0, var_7)
    assert var_20 == 'timeout_test'



# Parsed testcases at query #8
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_line_17_false. Retrieved 16/20 statements.


import requests.cookies as module_0

def test_case_0():
    var_0 = 200
    var_1 = module_0.MockResponse()
    var_2 = []
    var_3 = 'method'
    var_4 = 'timeout'
    var_5 = 'get'
    var_6 = 30
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'http://example.com'
    var_9 = 'get'
    var_10 = {}
    var_11 = None
    var_12 = None
    var_13 = var_1.status_code
    var_14 = 300
    var_15 = var_0 <= var_13 < var_14



# Parsed testcases at query #10
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_line9_false_due_to_method_not_get. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'test'
    var_4 = {var_2: var_3}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_false_when_method_not_string. Retrieved 3/4 statements.
# Partially parsed test_predicate_false_when_method_not_get. Retrieved 5/6 statements.
# Partially parsed test_predicate_false_when_data_is_none. Retrieved 3/4 statements.
# Partially parsed test_predicate_false_when_data_empty_string. Retrieved 5/6 statements.
# Partially parsed test_predicate_false_when_method_case_mismatch. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'post'
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = ''
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_get_method_with_data_dict_adds_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_list_adds_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_data_tuple_adds_to_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_url_has_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_url_ends_with_question_mark. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_url_ends_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_without_data_does_not_change_url. Retrieved 3/4 statements.
# Partially parsed test_query_post_method_with_data_encodes_data. Retrieved 7/8 statements.
# Partially parsed test_query_post_method_with_string_data_encodes. Retrieved 5/6 statements.
# Partially parsed test_query_method_not_get_does_not_modify_url. Retrieved 7/8 statements.
# Partially parsed test_query_get_method_with_no_data_kwarg. Retrieved 3/4 statements.
# Partially parsed test_query_get_method_with_empty_dict_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?existing=true&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'rawstring'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'PUT'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_url_opener_with_requests_get. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_urllib_get. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_requests_post. Retrieved 15/16 statements.
# Partially parsed test_url_opener_with_urllib_post. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_requests_get_timeout. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_urllib_get_timeout. Retrieved 7/8 statements.
# Partially parsed test_url_opener_with_requests_session. Retrieved 11/12 statements.
# Partially parsed test_url_opener_with_urllib_no_method. Retrieved 5/6 statements.
# Partially parsed test_url_opener_with_requests_no_encoding. Retrieved 9/10 statements.
# Partially parsed test_url_opener_with_urllib_no_timeout. Retrieved 5/6 statements.


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = None
    var_6 = 'utf-8'
    var_7 = 30
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'data'
    var_5 = 'post'
    var_6 = None
    var_7 = 'utf-8'
    var_8 = 30
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_11}
    var_13 = 'http://example.com'
    var_14 = module_0.url_opener(var_13, var_12)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 30
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = None
    var_6 = 'utf-8'
    var_7 = 5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'encoding'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = None
    var_6 = 'utf-8'
    var_7 = 30
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 30
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = None
    var_5 = 30
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0.url_opener(var_3, var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}



# Parsed testcases at query #16
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://test.com'
    var_2 = 'timeout'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = module_0.url_opener(var_1, var_4)
    var_6 = {var_2: var_3}
    var_7 = module_0._urllib(var_1, var_6)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_query_with_data_dict_get_method_no_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_list_get_method_existing_query. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_tuple_get_method_existing_query_ends_with_question. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_dict_get_method_existing_query_ends_with_ampersand. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_dict_post_method. Retrieved 7/8 statements.
# Partially parsed test_query_with_data_none. Retrieved 3/4 statements.
# Partially parsed test_query_with_data_string. Retrieved 5/6 statements.
# Partially parsed test_query_with_data_dict_get_method_no_data. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?param=1'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com?a=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'b'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_false. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'Response'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 200
    var_4 = {var_2: var_3}



# Parsed testcases at query #19
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)



# Parsed testcases at query #20
#--------------------------




import requests.cookies as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 200
    var_5 = var_3
    var_6 = 'OK'
    var_7 = {}
    var_8 = 'requests'
    var_9 = __import__(var_8)
    var_10 = module_0.MockResponse()



# Parsed testcases at query #21
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_requests_with_session_get. Retrieved 6/11 statements.
# Partially parsed test_requests_without_session_get. Retrieved 5/7 statements.
# Partially parsed test_requests_with_encoding. Retrieved 7/11 statements.
# Partially parsed test_requests_http_error. Retrieved 5/7 statements.
# Partially parsed test_requests_with_data_in_get. Retrieved 9/12 statements.
# Partially parsed test_requests_with_session_post. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0._requests(var_3, var_2)
    assert var_4 == 'response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'response'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = module_0._requests(var_3, var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'key=value'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == 'response'
    var_7 = 'http://example.com?key=value'
    var_8 = 60

def test_case_0():
    var_0 = 'session'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'key=value'
    var_5 = 'http://example.com'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_line9_false_with_no_data. Retrieved 3/4 statements.
# Partially parsed test_predicate_line9_false_with_non_get_method. Retrieved 5/6 statements.
# Partially parsed test_predicate_line9_false_with_get_and_no_data. Retrieved 3/4 statements.
# Partially parsed test_predicate_line9_false_with_mixed_case_non_get. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'Get'
    var_2 = 'data'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #24
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)
    var_4 = {}
    var_5 = module_0._urllib(var_1, var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_status_code_in_range_does_not_raise. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'MockResponse'
    var_1 = ()
    var_2 = 'status_code'
    var_3 = 'url'
    var_4 = 'reason'
    var_5 = 'headers'
    var_6 = 200
    var_7 = 'http://example.com'
    var_8 = 'OK'
    var_9 = {}
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = 300



# Parsed testcases at query #26
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = False
    var_1 = 'http://example.com'
    var_2 = {}
    var_3 = module_0.url_opener(var_1, var_2)



# Parsed testcases at query #27
#--------------------------




import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key=value'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = {}
    var_3 = module_0._query(var_0, var_1, var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = ''
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 0
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)



