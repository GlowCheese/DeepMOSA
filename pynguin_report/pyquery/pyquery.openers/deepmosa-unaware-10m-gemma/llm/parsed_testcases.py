####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>success</html>'
    var_5 = 'http://example.com/404'
    var_6 = 'method'
    var_7 = 'get'
    var_8 = {var_6: var_7}
    var_9 = module_0.url_opener(var_5, var_8)
    var_10 = 'method'
    var_11 = 'data'
    var_12 = 'get'
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = {var_10: var_12, var_11: var_15}
    var_17 = 'http://example.com'
    var_18 = module_0.url_opener(var_17, var_16)
    var_19 = 'http://example.com'
    var_20 = 'method'
    var_21 = 'data'
    var_22 = 'get'
    var_23 = 'a'
    var_24 = 'b'
    var_25 = {var_23: var_24}
    var_26 = {var_20: var_22, var_21: var_25}
    var_27 = module_0.url_opener(var_19, var_26)
    var_28 = 'http://example.com?existing=true'
    var_29 = 'method'
    var_30 = 'data'
    var_31 = 'get'
    var_32 = 'new'
    var_33 = 'val'
    var_34 = {var_32: var_33}
    var_35 = {var_29: var_31, var_30: var_34}
    var_36 = 'http://example.com'
    var_37 = {var_32: var_33}
    var_38 = {var_29: var_31, var_30: var_37}
    var_39 = 'post'
    var_40 = 'key'
    var_41 = 'value'
    var_42 = {var_40: var_41}
    var_43 = {var_29: var_39, var_30: var_42}
    var_44 = 'http://example.com'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'http://example.com'

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
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'headers'
    var_2 = 'cookies'
    var_3 = 'timeout'
    var_4 = 'unrelated'
    var_5 = 'get'
    var_6 = 'User-Agent'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 'session'
    var_10 = '123'
    var_11 = {var_9: var_10}
    var_12 = 10
    var_13 = 'noise'
    var_14 = {var_0: var_5, var_1: var_8, var_2: var_11, var_3: var_12, var_4: var_13}
    var_15 = 'http://example.com'
    var_16 = module_0.url_opener(var_15, var_14)



# Parsed testcases at query #3
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>Success</html>'
    var_9 = 'http://example.com'
    var_10 = {}
    var_11 = module_0.url_opener(var_9, var_10)
    var_12 = 'method'
    var_13 = 'timeout'
    var_14 = 'GET'
    var_15 = 5
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 'http://example.com'
    var_18 = module_0.url_opener(var_17, var_16)
    var_19 = 'data'
    var_20 = 'method'
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 1
    var_24 = 2
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'post'
    var_27 = {var_19: var_25, var_20: var_26}
    var_28 = 'http://example.com'
    var_29 = module_0.url_opener(var_28, var_27)
    var_30 = 'headers'
    var_31 = 'auth'
    var_32 = 'User-Agent'
    var_33 = 'Test'
    var_34 = {var_32: var_33}
    var_35 = 'user'
    var_36 = 'pass'
    var_37 = (var_35, var_36)
    var_38 = {var_30: var_34, var_31: var_37}
    var_39 = 'http://example.com'
    var_40 = module_0.url_opener(var_39, var_38)



# Parsed testcases at query #4
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>Success</html>'
    var_9 = {var_3: var_4}
    var_10 = 'data'
    var_11 = 'method'
    var_12 = 'key'
    var_13 = 'val'
    var_14 = {var_12: var_13}
    var_15 = 'GET'
    var_16 = {var_10: var_14, var_11: var_15}
    var_17 = 'http://example.com'
    var_18 = module_0.url_opener(var_17, var_16)
    var_19 = 'http://example.com'
    var_20 = {}
    var_21 = module_0.url_opener(var_19, var_20)
    var_22 = 'method'
    var_23 = 'data'
    var_24 = 'POST'
    var_25 = 'a'
    var_26 = 'b'
    var_27 = {var_25: var_26}
    var_28 = {var_22: var_24, var_23: var_27}
    var_29 = 'http://example.com'
    var_30 = module_0.url_opener(var_29, var_28)
    var_31 = 'method'
    var_32 = 'data'
    var_33 = 'GET'
    var_34 = 'c'
    var_35 = 'd'
    var_36 = {var_34: var_35}
    var_37 = {var_31: var_33, var_32: var_36}
    var_38 = 'http://example.com?a=b'
    var_39 = module_0.url_opener(var_38, var_37)
    var_40 = 'encoding'
    var_41 = 'utf-16'
    var_42 = {var_40: var_41}
    var_43 = 'http://example.com'
    var_44 = module_0.url_opener(var_43, var_42)



# Parsed testcases at query #5
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'timeout'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = 'method'
    var_9 = 'data'
    var_10 = 'get'
    var_11 = 'a'
    var_12 = 1
    var_13 = {var_11: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = module_0.url_opener(var_7, var_14)
    assert var_15 == '<html>success</html>'
    var_16 = 'url'
    var_17 = 201
    var_18 = 'created'
    var_19 = 'post'
    var_20 = {var_2: var_3}
    var_21 = {var_8: var_19, var_9: var_20}
    var_22 = module_0.url_opener(var_7, var_21)
    assert var_22 == 'created'
    var_23 = 'http://example.com'
    var_24 = 'method'
    var_25 = 'get'
    var_26 = {var_24: var_25}
    var_27 = module_0.url_opener(var_23, var_26)
    var_28 = b'urllib_data'
    var_29 = lambda : var_28
    var_30 = 'http://example.com'
    var_31 = 'method'
    var_32 = 'data'
    var_33 = 'get'
    var_34 = 'test'
    var_35 = '1'
    var_36 = {var_34: var_35}
    var_37 = {var_31: var_33, var_32: var_36}
    var_38 = module_0.url_opener(var_30, var_37)
    var_39 = 'http://example.com'
    var_40 = 'method'
    var_41 = 'session'
    var_42 = 'get'
    var_43 = module_0.url_opener(var_39, var_32)
    var_44 = 'http://example.com'
    var_45 = 'method'
    var_46 = 'encoding'
    var_47 = 'get'
    var_48 = 'latin-1'
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = module_0.url_opener(var_44, var_49)



# Parsed testcases at query #6
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 'http://test.com'
    var_2 = 'utf-8'
    var_3 = module_0.encode(var_2)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>success</html>'

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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'headers'
    var_3 = 'timeout'
    var_4 = 'get'
    var_5 = 'query'
    var_6 = 'pytest'
    var_7 = {var_5: var_6}
    var_8 = 'User-Agent'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = 10
    var_12 = {var_0: var_4, var_1: var_7, var_2: var_10, var_3: var_11}
    var_13 = 'http://example.com'
    var_14 = module_0.url_opener(var_13, var_12)

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'session'
    var_2 = 'method'
    var_3 = 'get'



# Parsed testcases at query #7
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'success'
    var_5 = 'http://example.com'
    var_6 = 'method'
    var_7 = 'get'
    var_8 = {var_6: var_7}
    var_9 = module_0.url_opener(var_5, var_8)
    var_10 = b'urllib_data'
    var_11 = lambda : var_10
    var_12 = 'http://example.com'
    var_13 = 'method'
    var_14 = 'data'
    var_15 = 'get'
    var_16 = 'key'
    var_17 = 'val'
    var_18 = {var_16: var_17}
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = module_0.url_opener(var_12, var_19)
    var_21 = 0
    var_22 = 'http://example.com'
    var_23 = 'method'
    var_24 = 'data'
    var_25 = 'get'
    var_26 = 'a'
    var_27 = 1
    var_28 = {var_26: var_27}
    var_29 = {var_23: var_25, var_24: var_28}
    var_30 = module_0.url_opener(var_22, var_29)
    var_31 = 'url'
    var_32 = 'method'
    var_33 = 'headers'
    var_34 = 'unsupported_arg'
    var_35 = 'get'
    var_36 = 'User-Agent'
    var_37 = 'test'
    var_38 = {var_36: var_37}
    var_39 = 'ignore_me'
    var_40 = {var_32: var_35, var_33: var_38, var_34: var_39}
    var_41 = 'http://example.com'
    var_42 = module_0.url_opener(var_41, var_40)
    var_43 = 1
    var_44 = 'http://example.com'
    var_45 = 'method'
    var_46 = 'session'
    var_47 = 'get'
    var_48 = module_0.url_opener(var_44, var_36)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'method'
    var_1 = 'http://test.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'success'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'auth'
    var_3 = 'post'
    var_4 = 'key'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = 'u'
    var_8 = 'p'
    var_9 = (var_7, var_8)
    var_10 = {var_0: var_3, var_1: var_6, var_2: var_9}
    var_11 = 'http://test.com'
    var_12 = module_0.url_opener(var_11, var_10)
    assert var_12 == 'created'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'encoding'
    var_2 = 'utf-16'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)



# Parsed testcases at query #9
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'encoding'
    var_2 = 10
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0.url_opener(var_5, var_4)
    assert var_6 == 'success_html'
    var_7 = 'http://example.com'
    var_8 = {}
    var_9 = module_0.url_opener(var_7, var_8)
    var_10 = 'data'
    var_11 = 'method'
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 'get'
    var_16 = {var_10: var_14, var_11: var_15}
    var_17 = 'http://example.com'
    var_18 = module_0.url_opener(var_17, var_16)
    var_19 = 'http://example.com'
    var_20 = 'timeout'
    var_21 = 5
    var_22 = {var_20: var_21}
    var_23 = module_0.url_opener(var_19, var_22)
    var_24 = None
    var_25 = 'method'
    var_26 = 'data'
    var_27 = 'post'
    var_28 = 'a'
    var_29 = 'b'
    var_30 = {var_28: var_29}
    var_31 = {var_25: var_27, var_26: var_30}
    var_32 = 'http://example.com'
    var_33 = module_0.url_opener(var_32, var_31)
    var_34 = 'auth'
    var_35 = 'headers'
    var_36 = 'invalid_arg'
    var_37 = 'user'
    var_38 = 'pass'
    var_39 = (var_37, var_38)
    var_40 = 'User-Agent'
    var_41 = 'test'
    var_42 = {var_40: var_41}
    var_43 = 'should_be_ignored'
    var_44 = {var_34: var_39, var_35: var_42, var_36: var_43}
    var_45 = 'http://example.com'
    var_46 = module_0.url_opener(var_45, var_44)



# Parsed testcases at query #10
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<html>success</html>'
    var_3 = 60
    var_4 = 'data'
    var_5 = 'timeout'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 10
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = module_0.url_opener(var_0, var_10)
    var_12 = 'encoding'
    var_13 = 'utf-8'
    var_14 = {var_12: var_13}
    var_15 = module_0.url_opener(var_0, var_14)
    var_16 = 'http://example.com'
    var_17 = {}
    var_18 = module_0.url_opener(var_16, var_17)
    var_19 = 'http://example.com'
    var_20 = {}
    var_21 = module_0.url_opener(var_19, var_20)
    var_22 = None
    var_23 = 60
    var_24 = 'method'
    var_25 = 'data'
    var_26 = 'post'
    var_27 = 'a'
    var_28 = 'b'
    var_29 = {var_27: var_28}
    var_30 = {var_24: var_26, var_25: var_29}
    var_31 = module_0.url_opener(var_19, var_30)
    var_32 = 'session'
    var_33 = 'http://example.com'
    var_34 = module_0.url_opener(var_33, var_30)
    assert var_34 == 'session_data'
    var_35 = 'data'
    var_36 = 'headers'
    var_37 = 'unallowed_arg'
    var_38 = 'id'
    var_39 = '1'
    var_40 = {var_38: var_39}
    var_41 = 'User-Agent'
    var_42 = 'test'
    var_43 = {var_41: var_42}
    var_44 = 'should_be_ignored'
    var_45 = {var_35: var_40, var_36: var_43, var_37: var_44}
    var_46 = 'http://example.com'
    var_47 = module_0.url_opener(var_46, var_45)
    var_48 = 1



# Parsed testcases at query #11
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>Success</html>'
    var_9 = 'http://example.com/bad'
    var_10 = {}
    var_11 = module_0.url_opener(var_9, var_10)
    var_12 = 'http://example.com'
    var_13 = 'timeout'
    var_14 = 5
    var_15 = {var_13: var_14}
    var_16 = module_0.url_opener(var_12, var_15)
    var_17 = 'data'
    var_18 = 'method'
    var_19 = 'key'
    var_20 = 'val'
    var_21 = {var_19: var_20}
    var_22 = 'get'
    var_23 = {var_17: var_21, var_18: var_22}
    var_24 = 'http://example.com'
    var_25 = module_0.url_opener(var_24, var_23)
    var_26 = 'data'
    var_27 = 'method'
    var_28 = 'key'
    var_29 = 'val'
    var_30 = {var_28: var_29}
    var_31 = 'post'
    var_32 = {var_26: var_30, var_27: var_31}
    var_33 = 'http://example.com'
    var_34 = module_0.url_opener(var_33, var_32)
    var_35 = 'session'
    var_36 = 'method'
    var_37 = 'get'
    var_38 = 'http://example.com'
    var_39 = module_0.url_opener(var_38, var_32)
    assert var_39 == 'session_ok'



# Parsed testcases at query #12
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'params'
    var_1 = 'key'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://example.com'
    var_6 = 'data'
    var_7 = 'timeout'
    var_8 = 'a'
    var_9 = 1
    var_10 = {var_8: var_9}
    var_11 = 10
    var_12 = {var_6: var_10, var_7: var_11}
    var_13 = module_0.url_opener(var_5, var_12)
    assert var_13 == '<html>Success</html>'
    var_14 = 'http://example.com'
    var_15 = {}
    var_16 = module_0.url_opener(var_14, var_15)
    var_17 = 'http://example.com'
    var_18 = 'data'
    var_19 = 'method'
    var_20 = 'key'
    var_21 = 'val'
    var_22 = {var_20: var_21}
    var_23 = 'post'
    var_24 = {var_18: var_22, var_19: var_23}
    var_25 = module_0.url_opener(var_17, var_24)
    var_26 = 'http://test.com?existing=true'
    var_27 = 'get'
    var_28 = 'data'
    var_29 = 'new'
    var_30 = 'val'
    var_31 = {var_29: var_30}
    var_32 = {var_28: var_31}
    var_33 = 'data'
    var_34 = 'auth'
    var_35 = 'invalid_arg'
    var_36 = 'foo'
    var_37 = 'bar'
    var_38 = {var_36: var_37}
    var_39 = 'user'
    var_40 = 'pass'
    var_41 = (var_39, var_40)
    var_42 = 'should_be_ignored'
    var_43 = {var_33: var_38, var_34: var_41, var_35: var_42}
    var_44 = 'http://example.com'
    var_45 = module_0.url_opener(var_44, var_43)
    var_46 = 'http://example.com'
    var_47 = 'encoding'
    var_48 = 'utf-16'
    var_49 = {var_47: var_48}
    var_50 = module_0.url_opener(var_46, var_49)



# Parsed testcases at query #13
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>Success</html>'
    var_9 = 'data'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = 'http://example.com'
    var_15 = module_0.url_opener(var_14, var_13)
    assert var_15 == 'data_result'



# Parsed testcases at query #14
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>Success</html>'
    var_5 = 'data'
    var_6 = 'method'
    var_7 = 'key'
    var_8 = 'val'
    var_9 = {var_7: var_8}
    var_10 = 'get'
    var_11 = {var_5: var_9, var_6: var_10}
    var_12 = 'http://example.com'
    var_13 = module_0.url_opener(var_12, var_11)
    var_14 = 'http://example.com'
    var_15 = 'method'
    var_16 = 'get'
    var_17 = {var_15: var_16}
    var_18 = module_0.url_opener(var_14, var_17)
    assert var_18 == b'urllib content'
    var_19 = 'http://example.com'
    var_20 = 'method'
    var_21 = 'get'
    var_22 = {var_20: var_21}
    var_23 = module_0.url_opener(var_19, var_22)
    var_24 = 'http://example.com?existing=1'
    var_25 = 'data'
    var_26 = 'method'
    var_27 = 'new'
    var_28 = '2'
    var_29 = {var_27: var_28}
    var_30 = 'get'
    var_31 = {var_25: var_29, var_26: var_30}
    var_32 = module_0.url_opener(var_24, var_31)
    var_33 = 'headers'
    var_34 = 'timeout'
    var_35 = 'User-Agent'
    var_36 = 'Test'
    var_37 = {var_35: var_36}
    var_38 = 10
    var_39 = {var_33: var_37, var_34: var_38}
    var_40 = 'http://example.com'
    var_41 = module_0.url_opener(var_40, var_39)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import requests.api as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 'method'
    var_2 = module_0.get(var_0)
    var_3 = {}
    var_4 = 'data'
    var_5 = 'http://example.com'

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
    var_3 = 'get'
    var_4 = 'k'
    var_5 = 'v'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'headers'
    var_2 = 'cookies'
    var_3 = 'post'
    var_4 = 'User-Agent'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'session'
    var_8 = '123'
    var_9 = {var_7: var_8}
    var_10 = {var_0: var_3, var_1: var_6, var_2: var_9}
    var_11 = 'http://example.com'
    var_12 = module_0.url_opener(var_11, var_10)



# Parsed testcases at query #2
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 'timeout'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>success</html>'
    var_9 = 'http://example.com'
    var_10 = {}
    var_11 = module_0.url_opener(var_9, var_10)
    var_12 = 'method'
    var_13 = 'timeout'
    var_14 = 'get'
    var_15 = 5
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 'http://example.com'
    var_18 = module_0.url_opener(var_17, var_16)
    var_19 = 'http://example.com?a=b'
    var_20 = 'data'
    var_21 = 'method'
    var_22 = 'c'
    var_23 = 'd'
    var_24 = {var_22: var_23}
    var_25 = 'get'
    var_26 = {var_20: var_24, var_21: var_25}
    var_27 = module_0.url_opener(var_19, var_26)
    var_28 = 'session'
    var_29 = 'method'
    var_30 = 'get'
    var_31 = 'http://example.com'
    var_32 = module_0.url_opener(var_31, var_16)
    assert var_32 == 'session data'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'method'
    var_1 = 'http://test.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>success</html>'

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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 'method'
    var_5 = 'data'
    var_6 = 'post'
    var_7 = {var_4: var_6, var_5: var_2}
    var_8 = module_0.url_opener(var_3, var_7)
    assert var_8 == 'Created'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'headers'
    var_2 = 'timeout'
    var_3 = 'extra_arg'
    var_4 = 'get'
    var_5 = 'User-Agent'
    var_6 = 'Test'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = 'ignore_me'
    var_10 = {var_0: var_4, var_1: var_7, var_2: var_8, var_3: var_9}
    var_11 = 'http://example.com'
    var_12 = module_0.url_opener(var_11, var_10)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'http://test.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>success</html>'

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

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 'method'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = 'post'
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == 'Created'
    var_9 = b'key=val'

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'session'
    var_2 = 'method'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'data'
    var_1 = 'http://test.com/path'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://test.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'POST'
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://test.com'
    var_8 = module_0.url_opener(var_7, var_6)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'content'

def test_case_0():
    var_0 = 'session'
    var_1 = 'http://test.com'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'data'
    var_1 = 'http://test.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

def test_case_0():
    var_0 = 'data'
    var_1 = 'b'
    var_2 = '2'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'http://test.com?a=1'
    var_6 = 'get'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'http://test.com'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'timeout'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'id'
    var_5 = '123'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    var_9 = 'url'
    var_10 = 1

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://test.com'
    var_6 = module_0.url_opener(var_5, var_4)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://test.com'
    var_6 = module_0.url_opener(var_5, var_4)

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://test.com'

def test_case_0():
    var_0 = 'http://example.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}

def test_case_0():
    var_0 = 'http://example.com?'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'GET'
    var_4 = 'new'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://test.com'
    var_6 = module_0.url_opener(var_5, var_4)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'headers'
    var_2 = 'auth'
    var_3 = 'get'
    var_4 = 'User-Agent'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'u'
    var_8 = 'p'
    var_9 = (var_7, var_8)
    var_10 = {var_0: var_3, var_1: var_6, var_2: var_9}
    var_11 = 'http://test.com'
    var_12 = module_0.url_opener(var_11, var_10)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 10

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
    var_9 = module_0.url_opener(var_0, var_7)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com/404'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    var_5 = module_0.url_opener(var_0, var_3)

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
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'headers'
    var_3 = 'extra'
    var_4 = 'get'
    var_5 = 'User-Agent'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = 'ignore'
    var_9 = {var_1: var_4, var_2: var_7, var_3: var_8}
    var_10 = module_0.url_opener(var_0, var_9)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'http://test.com'
    var_3 = 0
    var_4 = 1

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == '<html>success</html>'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'post'
    var_4 = 'id'
    var_5 = '1'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)

def test_case_0():
    var_0 = 'http://test.com'
    var_1 = 'method'
    var_2 = 'session'
    var_3 = 'get'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://test.com?existing=true'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = 'new'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_0.url_opener(var_0, var_7)
    var_9 = 0



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'http://test.com/path'
    var_3 = 'http://test.com/path?'
    var_4 = 'http://test.com/path&'

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'timeout'
    var_3 = 'get'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)

import pyquery.openers as module_0

def test_case_0():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'headers'
    var_3 = 'get'
    var_4 = 5
    var_5 = 'User-Agent'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_7}
    var_9 = 'http://example.com'
    var_10 = module_0.url_opener(var_9, var_8)
    assert var_10 == '<html></html>'

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
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == 'content'

def test_case_0():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = 'get'
    var_3 = 'http://example.com'



# Parsed testcases at query #12
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'utf-8'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == '<html>Success</html>'
    var_7 = 'http://example.com'
    var_8 = 'method'
    var_9 = 'get'
    var_10 = {var_8: var_9}
    var_11 = module_0.url_opener(var_7, var_10)
    var_12 = 'http://example.com'
    var_13 = 'method'
    var_14 = 'data'
    var_15 = 'get'
    var_16 = 'key'
    var_17 = 'val'
    var_18 = {var_16: var_17}
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = module_0.url_opener(var_12, var_19)
    var_21 = 'http://example.com'
    var_22 = 'method'
    var_23 = 'get'
    var_24 = {var_22: var_23}
    var_25 = module_0.url_opener(var_21, var_24)
    var_26 = 'http://example.com'
    var_27 = 'method'
    var_28 = 'data'
    var_29 = 'post'
    var_30 = 'a'
    var_31 = 'b'
    var_32 = {var_30: var_31}
    var_33 = {var_27: var_29, var_28: var_32}
    var_34 = module_0.url_opener(var_26, var_33)
    var_35 = 'http://example.com'
    var_36 = 'method'
    var_37 = 'session'
    var_38 = 'get'
    var_39 = module_0.url_opener(var_35, var_30)
    assert var_39 == 'session_data'



# Parsed testcases at query #13
#--------------------------


import pyquery.openers as module_0

def test_case_0():
    var_0 = 'timeout'
    var_1 = 'data'
    var_2 = 10
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    var_8 = module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>Success</html>'
    var_9 = 'http://example.com?key=val'
    var_10 = None
    var_11 = 'http://example.com'
    var_12 = 'method'
    var_13 = 'get'
    var_14 = {var_12: var_13}
    var_15 = module_0.url_opener(var_11, var_14)
    var_16 = b'urllib data'
    var_17 = lambda : var_16
    var_18 = 'http://example.com'
    var_19 = 'method'
    var_20 = 'data'
    var_21 = 'get'
    var_22 = 'a'
    var_23 = '1'
    var_24 = {var_22: var_23}
    var_25 = {var_19: var_21, var_20: var_24}
    var_26 = module_0.url_opener(var_18, var_25)
    var_27 = 0
    var_28 = 'method'
    var_29 = 'headers'
    var_30 = 'encoding'
    var_31 = 'post'
    var_32 = 'foo'
    var_33 = 'bar'
    var_34 = {var_32: var_33}
    var_35 = 'User-Agent'
    var_36 = 'test'
    var_37 = {var_35: var_36}
    var_38 = 'utf-8'
    var_39 = {var_28: var_31, var_17: var_34, var_29: var_37, var_30: var_38}
    var_40 = 'http://example.com/post'
    var_41 = module_0.url_opener(var_40, var_39)
    assert var_41 == 'Created'
    var_42 = 'http://example.com'
    var_43 = 'get'
    var_44 = {}
    var_45 = 'http://example.com?existing=true'
    var_46 = 'new'
    var_47 = 'val'
    var_48 = {var_46: var_47}
    var_49 = 'post'
    var_50 = 'key'
    var_51 = 'value'
    var_52 = {var_50: var_51}



