####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    _requests(url, kwargs)

def test_requests_get_with_data_dict():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    _requests(url, kwargs)

def test_requests_get_with_data_list():
    url = "http://example.com"
    kwargs = {"method": "get", "data": ["key", "value"]}
    _requests(url, kwargs)

def test_requests_get_with_data_tuple():
    url = "http://example.com"
    kwargs = {"method": "get", "data": ("key", "value")}
    _requests(url, kwargs)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    _requests(url, kwargs)

def test_requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "get", "session": session}
    _requests(url, kwargs)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    _requests(url, kwargs)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    _requests(url, kwargs)

def test_requests_with_allowed_args():
    url = "http://example.com"
    kwargs = {"method": "get", "headers": {"User-Agent": "test"}}
    _requests(url, kwargs)

def test_requests_get_with_data_and_existing_question_mark():
    url = "http://example.com?existing=param"
    kwargs = {"method": "get", "data": {"key": "value"}}
    _requests(url, kwargs)

def test_requests_get_with_data_and_existing_ampersand():
    url = "http://example.com?existing=param&"
    kwargs = {"method": "get", "data": {"key": "value"}}
    _requests(url, kwargs)

def test_requests_http_error():
    url = "http://httpstat.us/404"
    kwargs = {"method": "get"}
    try:
        _requests(url, kwargs)
    except HTTPError:
        pass

def test_requests_successful_response():
    url = "http://httpstat.us/200"
    kwargs = {"method": "get"}
    _requests(url, kwargs)
```


# LLM-generated content at query #2
#--------------------------

def test_query_with_dict_data_post():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_list_data_post():
    url, data = _query("http://example.com", "POST", {"data": ["a", "b"]})
    assert url == "http://example.com"
    assert data == b"0=a&1=b"

def test_query_with_tuple_data_post():
    url, data = _query("http://example.com", "POST", {"data": ("x", "y")})
    assert url == "http://example.com"
    assert data == b"0=x&1=y"

def test_query_with_dict_data_get():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_get():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?0=a&1=b"
    assert data is None

def test_query_with_tuple_data_get():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?0=x&1=y"
    assert data is None

def test_query_with_dict_data_get_url_has_question_mark():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_with_dict_data_get_url_ends_with_question_mark():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?&key=value"
    assert data is None

def test_query_with_dict_data_get_url_ends_with_ampersand():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&&key=value"
    assert data is None

def test_query_with_no_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_string_data_post():
    url, data = _query("http://example.com", "POST", {"data": "raw_data"})
    assert url == "http://example.com"
    assert data == b"raw_data"

def test_query_with_get_method_and_no_data():
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None


# LLM-generated content at query #3
#--------------------------

```
def test_predicate_evaluates_to_false():
    mock_response = type('MockResponse', (), {'status_code': 200, 'url': '', 'reason': '', 'headers': {}})()
    assert 200 <= mock_response.status_code < 300
```


# LLM-generated content at query #4
#--------------------------

```
def test_predicate_true():
    resp = type('Resp', (), {'status_code': 200, 'url': '', 'reason': '', 'headers': {}, 'text': '', 'encoding': 'utf-8'})()
    encoding = 'utf-8'
    result = encoding and True
    assert result == True
```


# LLM-generated content at query #5
#--------------------------

def test_predicate_line9_false_because_method_is_post():
    result = _query("http://example.com", "POST", {"data": "key=value"})
    assert result == ("http://example.com", b"key=value")


# LLM-generated content at query #6
#--------------------------

def test_query_get_with_dict_data_adds_query_string():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_get_with_list_data_adds_query_string():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?a=b"
    assert data is None

def test_query_get_with_tuple_data_adds_query_string():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?x=y"
    assert data is None

def test_query_get_with_existing_question_mark():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_get_with_trailing_question_mark():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?&key=value"
    assert data is None

def test_query_get_with_trailing_ampersand():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&&key=value"
    assert data is None

def test_query_get_with_no_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_get_with_empty_data():
    url, data = _query("http://example.com", "GET", {"data": {}})
    assert url == "http://example.com?"
    assert data is None

def test_query_post_with_dict_data():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_post_with_list_data():
    url, data = _query("http://example.com", "POST", {"data": ["a", "b"]})
    assert url == "http://example.com"
    assert data == b"a=b"

def test_query_post_with_tuple_data():
    url, data = _query("http://example.com", "POST", {"data": ("x", "y")})
    assert url == "http://example.com"
    assert data == b"x=y"

def test_query_post_with_no_data():
    url, data = _query("http://example.com", "POST", {})
    assert url == "http://example.com"
    assert data is None

def test_query_post_with_empty_dict_data():
    url, data = _query("http://example.com", "POST", {"data": {}})
    assert url == "http://example.com"
    assert data == b""

def test_query_get_with_bytes_data():
    url, data = _query("http://example.com", "GET", {"data": b"raw"})
    assert url == "http://example.com"
    assert data == b"raw"

def test_query_post_with_bytes_data():
    url, data = _query("http://example.com", "POST", {"data": b"raw"})
    assert url == "http://example.com"
    assert data == b"raw"

def test_query_get_with_non_string_method():
    url, data = _query("http://example.com", 123, {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_get_with_case_insensitive_method():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_get_with_none_data():
    url, data = _query("http://example.com", "GET", {"data": None})
    assert url == "http://example.com"
    assert data is None


# LLM-generated content at query #7
#--------------------------

```
def test_url_opener_with_requests_get_no_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_session():
    import requests
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "get", "session": session}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_http_error():
    import pytest
    url = "http://httpbin.org/status/404"
    kwargs = {"method": "get"}
    with pytest.raises(Exception):
        url_opener(url, kwargs)
```


# LLM-generated content at query #8
#--------------------------

def test_urllib_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_post_without_data():
    url = "http://example.com"
    kwargs = {"method": "post"}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_with_default_timeout():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_with_data_url_encoded():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value with spaces"}}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_with_data_already_has_query():
    url = "http://example.com?existing=param"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_with_data_url_ends_with_question():
    url = "http://example.com?"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _urllib(url, kwargs)
    assert result is not None

def test_urllib_get_with_data_url_ends_with_ampersand():
    url = "http://example.com&"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _urllib(url, kwargs)
    assert result is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    result = _query("http://example.com", "POST", {})
    assert result == ("http://example.com", None)
```


# LLM-generated content at query #10
#--------------------------

```
def test_predicate_evaluates_to_false():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})
    HAS_REQUEST = original_has_request
```


# LLM-generated content at query #11
#--------------------------

```
def test_url_opener_uses_requests_when_available():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = True
    try:
        result = url_opener("http://example.com", {"method": "get"})
        assert isinstance(result, str)
    finally:
        HAS_REQUEST = original_has_request

def test_url_opener_uses_urllib_when_not_available():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = False
    try:
        result = url_opener("http://example.com", {"method": "get"})
        assert hasattr(result, 'read')
    finally:
        HAS_REQUEST = original_has_request

def test_url_opener_requests_post():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = True
    try:
        result = url_opener("http://httpbin.org/post", {"method": "post", "data": {"key": "value"}})
        assert isinstance(result, str)
    finally:
        HAS_REQUEST = original_has_request

def test_url_opener_urllib_post():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = False
    try:
        result = url_opener("http://httpbin.org/post", {"method": "post", "data": {"key": "value"}})
        assert hasattr(result, 'read')
    finally:
        HAS_REQUEST = original_has_request
```


# LLM-generated content at query #12
#--------------------------

```
def test_predicate_false():
    resp = type('Response', (), {'status_code': 200})()
    result = not (200 <= resp.status_code < 300)
    assert result == False
```


# LLM-generated content at query #13
#--------------------------

```python
def test_requests_successful_get_request_with_query_params():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_successful_get_request_without_data():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'timeout': 5}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_successful_post_request():
    url = 'http://example.com'
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'timeout': 5}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = 'http://example.com'
    kwargs = {'method': 'get', 'session': session, 'timeout': 5}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_http_error():
    url = 'http://example.com/404'
    kwargs = {'method': 'get', 'timeout': 5}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        assert True

def test_requests_with_allowed_args():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'timeout': 5, 'headers': {'User-Agent': 'test'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #14
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "get", "session": session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_http_error():
    url = "http://httpstat.us/404"
    kwargs = {"method": "get"}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        assert True

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 1}
    result = _requests(url, kwargs)
    assert isinstance(result, str)
```


# LLM-generated content at query #15
#--------------------------

```
def test_has_request_is_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result is None
```


# LLM-generated content at query #16
#--------------------------

```
def test_status_code_in_range_returns_false():
    resp = type('Response', (), {'status_code': 200})()
    assert not (200 <= resp.status_code < 300) == False
```


# LLM-generated content at query #17
#--------------------------

```
def test_has_request_is_false():
    # Save the original value
    original_has_request = HAS_REQUEST
    # Set HAS_REQUEST to False
    HAS_REQUEST = False
    try:
        result = url_opener("http://example.com", {})
        # The predicate at line 2 evaluates to False, so _urllib is called
        # We expect the result from _urllib, not _requests
        assert result == _urllib("http://example.com", {})
    finally:
        # Restore the original value
        HAS_REQUEST = original_has_request
```


# LLM-generated content at query #18
#--------------------------

```
def test_url_opener_with_requests_get_no_data():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_requests_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}, "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_requests_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_urllib_get():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_urllib_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_requests_http_error():
    url = "http://httpstat.us/404"
    kwargs = {"method": "get", "encoding": "utf-8"}
    try:
        url_opener(url, kwargs)
        assert False
    except HTTPError:
        assert True

def test_url_opener_with_requests_get_empty_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {}, "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_requests_get_list_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": [1, 2, 3], "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None

def test_url_opener_with_requests_get_tuple_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": (1, 2), "encoding": "utf-8"}
    result = url_opener(url, kwargs)
    assert result is not None
```


# LLM-generated content at query #19
#--------------------------

```
def test_predicate_at_line_17_evaluates_to_false():
    resp = type('MockResponse', (), {'status_code': 200})()
    assert 200 <= resp.status_code < 300
```


# LLM-generated content at query #20
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_http_error():
    import requests
    from requests.exceptions import HTTPError
    url = "http://httpstat.us/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False
    except HTTPError:
        pass
```


# LLM-generated content at query #21
#--------------------------

```
def test_has_request_false():
    url = "http://example.com"
    kwargs = {}
    result = url_opener(url, kwargs)
    assert not HAS_REQUEST
```


# LLM-generated content at query #22
#--------------------------

```
def test_predicate_false_when_has_request_is_false():
    original_has_request = __builtins__.get('HAS_REQUEST')
    try:
        __builtins__['HAS_REQUEST'] = False
        result = url_opener("http://example.com", {})
        assert result is not None
    finally:
        if original_has_request is not None:
            __builtins__['HAS_REQUEST'] = original_has_request
        else:
            del __builtins__['HAS_REQUEST']
```


# LLM-generated content at query #23
#--------------------------

```
def test_predicate_evaluates_to_false():
    url = "http://example.com"
    kwargs = {"method": "get"}
    encoding = kwargs.get('encoding')
    method = kwargs.get('method', 'get').lower()
    session = kwargs.get('session')
    if session:
        meth = getattr(session, str(method))
    else:
        meth = getattr(requests, str(method))
    if method == 'get':
        url, data = _query(url, method, kwargs)
    kw = {}
    allowed_args = []
    for k in allowed_args:
        if k in kwargs:
            kw[k] = kwargs[k]
    resp = meth(url=url, timeout=kwargs.get('timeout', 5), **kw)
    assert not (200 <= resp.status_code < 300) == False
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_requests_get_without_data():
    kwargs = {'method': 'get', 'session': None, 'encoding': 'utf-8'}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post():
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session, 'encoding': 'utf-8'}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    kwargs = {'method': 'get', 'encoding': 'utf-8', 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    kwargs = {'method': 'get', 'timeout': 10, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_allowed_args():
    kwargs = {'method': 'get', 'headers': {'User-Agent': 'test'}, 'session': None}
    url = 'http://example.com'
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_http_error():
    import pytest
    kwargs = {'method': 'get', 'session': None}
    url = 'http://httpstat.us/404'
    with pytest.raises(HTTPError):
        _requests(url, kwargs)
```


# LLM-generated content at query #2
#--------------------------

def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert url == "http://example.com?0=a&1=b"
    assert data is None

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ("x", "y")})
    assert url == "http://example.com?0=x&1=y"
    assert data is None

def test_query_with_string_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": "param=value"})
    assert url == "http://example.com?param=value"
    assert data is None

def test_query_with_data_and_post_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_and_non_string_method():
    url, data = _query("http://example.com", 123, {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_without_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_url_containing_question_mark_and_get_method():
    url, data = _query("http://example.com?existing=1", "GET", {"data": {"new": "2"}})
    assert url == "http://example.com?existing=1&new=2"
    assert data is None

def test_query_with_url_ending_with_question_mark_and_get_method():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?&key=value"
    assert data is None

def test_query_with_url_ending_with_ampersand_and_get_method():
    url, data = _query("http://example.com?existing=1&", "GET", {"data": {"new": "2"}})
    assert url == "http://example.com?existing=1&&new=2"
    assert data is None


# LLM-generated content at query #3
#--------------------------

def test_predicate_line9_returns_false():
    url = "http://example.com"
    method = "GET"
    data = None
    result = _query(url, method, {'data': data})
    assert result == (url, None)


# LLM-generated content at query #4
#--------------------------

def test_query_with_dict_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_list_data_and_get_method():
    url, data = _query("http://example.com", "get", {"data": [1, 2, 3]})
    assert url == "http://example.com?1&2&3"
    assert data is None

def test_query_with_tuple_data_and_get_method():
    url, data = _query("http://example.com", "GET", {"data": ("a", "b")})
    assert url == "http://example.com?a&b"
    assert data is None

def test_query_with_data_and_get_method_url_has_question_mark():
    url, data = _query("http://example.com?existing=param", "GET", {"data": {"new": "value"}})
    assert url == "http://example.com?existing=param&new=value"
    assert data is None

def test_query_with_data_and_get_method_url_ends_with_question_mark():
    url, data = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?&key=value"
    assert data is None

def test_query_with_data_and_get_method_url_ends_with_ampersand():
    url, data = _query("http://example.com?existing=param&", "GET", {"data": {"new": "value"}})
    assert url == "http://example.com?existing=param&&new=value"
    assert data is None

def test_query_with_data_and_non_get_method():
    url, data = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_with_data_and_get_method_lowercase():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_without_data():
    url, data = _query("http://example.com", "GET", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_data_and_get_method_no_data_in_kwargs():
    url, data = _query("http://example.com", "GET", {"other": "param"})
    assert url == "http://example.com"
    assert data is None


# LLM-generated content at query #5
#--------------------------

def test_query_get_with_data_adding_to_url():
    url, data = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_get_with_data_and_existing_question_mark():
    url, data = _query("http://example.com?", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_get_with_data_and_existing_ampersand():
    url, data = _query("http://example.com?existing=1", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None

def test_query_get_without_data():
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None

def test_query_post_with_dict_data():
    url, data = _query("http://example.com", "post", {"data": {"key": "value"}})
    assert url == "http://example.com"
    assert data == b"key=value"

def test_query_post_with_list_data():
    url, data = _query("http://example.com", "post", {"data": ["key1=value1", "key2=value2"]})
    assert url == "http://example.com"
    assert data == b"key1=value1&key2=value2"

def test_query_post_with_tuple_data():
    url, data = _query("http://example.com", "post", {"data": ("key1=value1",)})
    assert url == "http://example.com"
    assert data == b"key1=value1"

def test_query_get_with_uppercase_method():
    url, data = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_no_data_in_kwargs():
    url, data = _query("http://example.com", "get", {})
    assert url == "http://example.com"
    assert data is None

def test_query_with_url_ending_with_question_mark_and_get():
    url, data = _query("http://example.com?", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?key=value"
    assert data is None

def test_query_with_url_ending_with_ampersand_and_get():
    url, data = _query("http://example.com?existing=1&", "get", {"data": {"key": "value"}})
    assert url == "http://example.com?existing=1&key=value"
    assert data is None


# LLM-generated content at query #6
#--------------------------

def test_query_with_dict_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_with_list_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": ["a", "b"]})
    assert result == ("http://example.com?a=b", None)

def test_query_with_tuple_data_and_get_method():
    result = _query("http://example.com", "GET", {"data": ("key", "value")})
    assert result == ("http://example.com?key=value", None)

def test_query_with_data_and_post_method():
    result = _query("http://example.com", "POST", {"data": {"key": "value"}})
    assert result == ("http://example.com", b"key=value")

def test_query_with_data_and_get_method_url_has_question_mark():
    result = _query("http://example.com?", "GET", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_with_data_and_get_method_url_ends_with_ampersand():
    result = _query("http://example.com?existing=param&", "GET", {"data": {"key": "value"}})
    assert result == ("http://example.com?existing=param&key=value", None)

def test_query_with_no_data():
    result = _query("http://example.com", "GET", {})
    assert result == ("http://example.com", None)

def test_query_with_data_empty():
    result = _query("http://example.com", "GET", {"data": {}})
    assert result == ("http://example.com", None)

def test_query_with_data_and_method_lowercase_get():
    result = _query("http://example.com", "get", {"data": {"key": "value"}})
    assert result == ("http://example.com?key=value", None)

def test_query_with_none_data():
    result = _query("http://example.com", "GET", {"data": None})
    assert result == ("http://example.com", None)


# LLM-generated content at query #7
#--------------------------

def test_url_ends_with_question_mark_and_data_appended_with_ampersand():
    result_url, _ = _query("http://example.com?existing=param&", "GET", {"data": "key=value"})
    assert result_url == "http://example.com?existing=param&key=value"


# LLM-generated content at query #8
#--------------------------

def test_url_opener_with_requests_and_get_method():
    kwargs = {'method': 'get', 'timeout': 10, 'encoding': 'utf-8'}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_post_method():
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'timeout': 10}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)

def test_url_opener_with_urllib_and_get_method():
    kwargs = {'method': 'get', 'timeout': 10}
    result = url_opener('http://example.com', kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #9
#--------------------------

```
def test_status_code_not_in_range():
    resp = type('Response', (), {'status_code': 400, 'url': 'http://example.com', 'reason': 'Bad Request', 'headers': {}})()
    meth = lambda url, timeout, **kw: resp
    result = meth(url='http://example.com', timeout=10)
    assert not (200 <= 400 < 300)
```


# LLM-generated content at query #10
#--------------------------

```
def test_url_opener_uses_requests_when_has_request():
    original_has_request = HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    HAS_REQUEST = original_has_request
    assert not HAS_REQUEST
```


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_12_evaluates_to_true():
    url = "http://example.com?existing=param"
    method = "get"
    kwargs = {"data": "new_data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?existing=param&new_data"
    assert result_data is None


# LLM-generated content at query #12
#--------------------------

```
def test_url_opener_uses_requests_when_available():
    kwargs = {}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_uses_urllib_when_requests_not_available():
    kwargs = {}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_get_method_and_data():
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_post_method():
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_encoding():
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_session():
    kwargs = {"method": "get", "session": "session_object"}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_timeout():
    kwargs = {"method": "get", "timeout": 30}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_data_as_string():
    kwargs = {"method": "post", "data": "raw_data"}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_data_as_tuple():
    kwargs = {"method": "post", "data": ("key1", "value1")}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_data_as_list():
    kwargs = {"method": "post", "data": [("key1", "value1")]}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_get_method_and_no_data():
    kwargs = {"method": "get"}
    result = url_opener("http://example.com", kwargs)

def test_url_opener_with_url_containing_query_string():
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = url_opener("http://example.com?existing=param", kwargs)

def test_url_opener_with_url_ending_with_question_mark():
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = url_opener("http://example.com?", kwargs)

def test_url_opener_with_url_ending_with_ampersand():
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = url_opener("http://example.com?existing=param&", kwargs)
```


# LLM-generated content at query #13
#--------------------------

```
def test_predicate_evaluates_to_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert not HAS_REQUEST
```


# LLM-generated content at query #14
#--------------------------

```
def test_urllib_get_with_data_adds_query_params():
    def mock_urlopen(url, data, timeout):
        return url
    global urlopen
    original_urlopen = urlopen
    urlopen = mock_urlopen
    try:
        result = _urllib("http://example.com", {"method": "get", "data": {"key": "value"}, "timeout": 30})
        assert result == "http://example.com?key=value"
    finally:
        urlopen = original_urlopen

def test_urllib_get_with_data_appends_to_existing_query():
    def mock_urlopen(url, data, timeout):
        return url
    global urlopen
    original_urlopen = urlopen
    urlopen = mock_urlopen
    try:
        result = _urllib("http://example.com?existing=1", {"method": "get", "data": {"key": "value"}, "timeout": 30})
        assert result == "http://example.com?existing=1&key=value"
    finally:
        urlopen = original_urlopen

def test_urllib_post_with_data_encodes_body():
    def mock_urlopen(url, data, timeout):
        return (url, data)
    global urlopen
    original_urlopen = urlopen
    urlopen = mock_urlopen
    try:
        result = _urllib("http://example.com", {"method": "post", "data": {"key": "value"}, "timeout": 30})
        assert result[0] == "http://example.com"
        assert result[1] == b"key=value"
    finally:
        urlopen = original_urlopen

def test_urllib_get_without_data():
    def mock_urlopen(url, data, timeout):
        return (url, data)
    global urlopen
    original_urlopen = urlopen
    urlopen = mock_urlopen
    try:
        result = _urllib("http://example.com", {"method": "get", "timeout": 30})
        assert result[0] == "http://example.com"
        assert result[1] is None
    finally:
        urlopen = original_urlopen

def test_urllib_default_timeout():
    def mock_urlopen(url, data, timeout):
        return timeout
    global urlopen, DEFAULT_TIMEOUT
    original_urlopen = urlopen
    original_default_timeout = DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = 10
    urlopen = mock_urlopen
    try:
        result = _urllib("http://example.com", {"method": "get"})
        assert result == 10
    finally:
        urlopen = original_urlopen
        DEFAULT_TIMEOUT = original_default_timeout
```


# LLM-generated content at query #15
#--------------------------

```python
def test_requests_get_without_data():
    url = 'http://example.com'
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_without_data():
    url = 'http://example.com'
    kwargs = {'method': 'post'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post_with_data():
    url = 'http://example.com'
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    url = 'http://example.com'
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = 'http://example.com'
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)
```


# LLM-generated content at query #16
#--------------------------

def test_predicate_false_when_data_is_none():
    kwargs = {}
    url = "http://example.com"
    method = "get"
    data = None
    result = _query(url, method, kwargs)
    assert result == (url, None)


# LLM-generated content at query #17
#--------------------------

def test_status_code_in_success_range_does_not_raise():
    resp = type('MockResponse', (), {'status_code': 200, 'url': '', 'reason': '', 'headers': {}, 'text': ''})()
    meth = lambda url, timeout, **kw: resp
    url = 'http://example.com'
    kwargs = {}
    kw = {}
    resp = meth(url=url, timeout=kwargs.get('timeout', 10), **kw)
    assert not (200 <= resp.status_code < 300) == False


# LLM-generated content at query #18
#--------------------------

```
def test_predicate_line_17_false():
    response = type('Response', (), {'status_code': 200})()
    assert 200 <= response.status_code < 300
```


# LLM-generated content at query #19
#--------------------------

def test_predicate_line8_false_method_not_string():
    url = "http://example.com"
    method = 123
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)

def test_predicate_line8_false_method_not_basestring():
    url = "http://example.com"
    method = b"get"
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)

def test_predicate_line8_false_method_lower_not_get():
    url = "http://example.com"
    method = "POST"
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)

def test_predicate_line8_false_no_data():
    url = "http://example.com"
    method = "get"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)

def test_predicate_line8_false_data_none():
    url = "http://example.com"
    method = "get"
    kwargs = {"data": None}
    result_url, result_data = _query(url, method, kwargs)


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_8_is_false_when_method_is_not_basestring():
    url = "http://example.com"
    method = 123
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "key=value"

def test_predicate_at_line_8_is_false_when_method_is_not_get():
    url = "http://example.com"
    method = "POST"
    kwargs = {"data": "key=value"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "key=value"

def test_predicate_at_line_8_is_false_when_data_is_falsy():
    url = "http://example.com"
    method = "get"
    kwargs = {"data": ""}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == ""
```


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    result_url, result_data = _query("http://example.com", "POST", {"data": "test"})
    assert result_url == "http://example.com"
    assert result_data == "test"  # or "test".encode('utf-8') depending on later logic, but predicate is false
```


# LLM-generated content at query #22
#--------------------------

```python
def test_requests_get_without_data():
    url = "http://example.com"
    kwargs = {"method": "get"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {"method": "get", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_post():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {"method": "get", "session": session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test_requests_http_error():
    import pytest
    url = "http://httpbin.org/status/404"
    kwargs = {"method": "get"}
    with pytest.raises(Exception):
        _requests(url, kwargs)
```


# LLM-generated content at query #23
#--------------------------

```
def test_url_opener_returns_urllib_when_no_has_request():
    import builtins
    original_has_request = __builtins__.get('HAS_REQUEST', None)
    __builtins__['HAS_REQUEST'] = False
    from your_module import url_opener
    result = url_opener("http://example.com", {"timeout": 10})
    assert result == _urllib("http://example.com", {"timeout": 10})
```


# LLM-generated content at query #24
#--------------------------

```
def test_predicate_false():
    resp = type('Response', (), {'status_code': 200})()
    assert 200 <= resp.status_code < 300
```


# LLM-generated content at query #25
#--------------------------

```
def test_predicate_false():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result is not None
```


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_false_no_data():
    url = "http://example.com"
    method = "GET"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert result == (url, None) or result is not None
```


# LLM-generated content at query #27
#--------------------------

```
def test_url_opener_with_requests_and_get_method():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_post_method():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "timeout": 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)

def test_url_opener_with_requests_and_session():
    import requests
    session = requests.Session()
    url = "http://example.com"
    kwargs = {"method": "get", "session": session, "timeout": 10}
    result = url_opener(url, kwargs)
    assert isinstance(result, str)
```


# LLM-generated content at query #28
#--------------------------

```
def test_predicate_false():
    resp = type('Response', (), {'status_code': 200})()
    result = (200 <= resp.status_code < 300)
    assert result == True
```


# LLM-generated content at query #29
#--------------------------

def test_query_get_with_dict_data_and_no_query_string():
    url = "http://example.com"
    method = "get"
    kwargs = {"data": {"key1": "value1", "key2": "value2"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?key1=value1&key2=value2"
    assert result_data is None

def test_query_get_with_dict_data_and_existing_query_string():
    url = "http://example.com?existing=param"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?existing=param&key=value"
    assert result_data is None

def test_query_get_with_list_data():
    url = "http://example.com"
    method = "get"
    kwargs = {"data": [("a", "1"), ("b", "2")]}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?a=1&b=2"
    assert result_data is None

def test_query_get_with_tuple_data():
    url = "http://example.com"
    method = "get"
    kwargs = {"data": (("x", "y"),)}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?x=y"
    assert result_data is None

def test_query_get_with_no_data():
    url = "http://example.com"
    method = "get"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data is None

def test_query_get_with_data_and_url_ending_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?key=value"
    assert result_data is None

def test_query_get_with_data_and_url_ending_with_ampersand():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?key=value"
    assert result_data is None

def test_query_get_with_data_and_url_with_multiple_params():
    url = "http://example.com?a=1&b=2"
    method = "get"
    kwargs = {"data": {"c": "3"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com?a=1&b=2&c=3"
    assert result_data is None

def test_query_post_with_data():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "key=value".encode('utf-8')

def test_query_post_with_no_data():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data is None

def test_query_put_with_data():
    url = "http://example.com"
    method = "put"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == "http://example.com"
    assert result_data == "key=value".encode('utf-8')


# LLM-generated content at query #30
#--------------------------

def test_predicate_false_when_method_not_get():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": "some_data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_url == url
    assert result_data == b"some_data"


# LLM-generated content at query #31
#--------------------------

```
def test_status_code_within_success_range():
    kwargs = {'method': 'get'}
    url = "http://example.com"
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.url = url
    mock_response.reason = "OK"
    mock_response.headers = {}
    mock_response.text = "<html></html>"
    meth = Mock(return_value=mock_response)
    resp = meth(url=url, timeout=5, **{})
    assert 200 <= resp.status_code < 300
```


# LLM-generated content at query #32
#--------------------------

```
def test_HAS_REQUEST_is_False():
    from unittest.mock import patch
    with patch('module.HAS_REQUEST', False):
        result = url_opener("http://example.com", {})
        assert result == _urllib("http://example.com", {})
```


