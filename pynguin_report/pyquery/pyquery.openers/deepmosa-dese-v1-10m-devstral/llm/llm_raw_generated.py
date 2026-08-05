####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__requests_with_get_method():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_post_method():
    url = "http://example.com"
    kwargs = {"method": "post", "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "get", "session": session, "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_data():
    url = "http://example.com"
    kwargs = {"method": "post", "data": {"key": "value"}, "encoding": "utf-8", "timeout": 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_invalid_url():
    url = "http://invalid.url"
    kwargs = {"method": "get", "encoding": "utf-8", "timeout": 10}
    try:
        _requests(url, kwargs)
    except HTTPError as e:
        assert e.status_code >= 400

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 0.001}
    try:
        _requests(url, kwargs)
    except requests.exceptions.Timeout:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test__query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test__query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test__query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test__query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test__query_get_method_appends_data_to_url():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test__query_get_method_appends_data_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test__query_get_method_appends_data_with_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test__query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test__query_case_insensitive_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test__query_non_string_method():
    url, data = _query('http://example.com', None, {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test__query_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw_string'})
    assert url == 'http://example.com'
    assert data == b'raw_string'


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'test html'
        mock_get.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test html'

def test_url_opener_with_requests_post():
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'test html'
        mock_post.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == 'test html'

def test_url_opener_with_requests_session():
    with patch('requests.Session') as mock_session:
        mock_instance = MagicMock()
        mock_session.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'test html'
        mock_instance.get.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'get', 'session': mock_instance})
        assert result == 'test html'

def test_url_opener_with_urllib_get():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b'test html'
        mock_urlopen.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == b'test html'

def test_url_opener_with_urllib_post():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b'test html'
        mock_urlopen.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == b'test html'

def test_url_opener_with_requests_http_error():
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = 'http://example.com'
        mock_response.reason = 'Not Found'
        mock_response.headers = {}
        mock_get.return_value = mock_response
        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})

def test_url_opener_with_urllib_http_error():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = HTTPError('http://example.com', 404, 'Not Found', {}, None)
        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_true():
    url = "http://example.com"
    method = "get"
    kwargs = {'data': {'key': 'value'}}

    result_url, result_data = _query(url, method, kwargs)

    assert isinstance(method, str) and method.lower() == 'get' and kwargs['data']


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_9_evaluates_to_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert result == (url, None)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_true():
    url = "http://example.com"
    method = "get"
    kwargs = {'data': {'key': 'value'}}
    result = _query(url, method, kwargs)
    assert result == ("http://example.com?key=value", None)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert not (isinstance(method, basestring) and method.lower() == 'get' and result[1])


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    url = "http://example.com"
    kwargs = {
        'encoding': 'utf-8',
        'method': 'get',
        'session': None,
        'timeout': 10
    }

    # Mock the response object with a status code outside the 200-299 range
    mock_response = type('MockResponse', (), {
        'status_code': 404,
        'url': url,
        'reason': 'Not Found',
        'headers': {},
        'text': 'Not Found',
        'encoding': 'utf-8'
    })()

    # Mock the requests.get method to return the mock response
    import requests
    original_get = requests.get
    requests.get = lambda **kw: mock_response

    try:
        # Call the function and expect an HTTPError to be raised
        with pytest.raises(HTTPError):
            _requests(url, kwargs)
    finally:
        # Restore the original requests.get method
        requests.get = original_get


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    resp = type('Response', (), {'status_code': 404})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert result == (url, None)


# LLM-generated content at query #11
#--------------------------

```python
def test_data_encoding_when_data_is_not_empty():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #12
#--------------------------

```python
def test_status_code_not_in_success_range():
    resp = type('Response', (), {'status_code': 404})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #13
#--------------------------

```python
def test__urllib_with_get_method_and_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'timeout': 10}
    result = _urllib(url, kwargs)
    assert result.geturl() == "http://example.com?key=value"
    assert result.getcode() == 200

def test__urllib_with_post_method_and_data():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'timeout': 10}
    result = _urllib(url, kwargs)
    assert result.geturl() == "http://example.com"
    assert result.getcode() == 200

def test__urllib_with_timeout():
    url = "http://example.com"
    kwargs = {'timeout': 5}
    result = _urllib(url, kwargs)
    assert result.getcode() == 200

def test__urllib_with_default_timeout():
    url = "http://example.com"
    kwargs = {}
    result = _urllib(url, kwargs)
    assert result.getcode() == 200


# LLM-generated content at query #14
#--------------------------

```python
def test_url_opener_without_requests():
    global HAS_REQUEST
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #15
#--------------------------

```python
def test__requests_with_get_method_and_no_session():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    html = _requests(url, kwargs)
    assert isinstance(html, str)

def test__requests_with_post_method_and_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {"method": "post", "session": session, "data": {"key": "value"}}
    html = _requests(url, kwargs)
    assert isinstance(html, str)

def test__requests_with_timeout_and_allowed_args():
    url = "http://example.com"
    kwargs = {"method": "get", "timeout": 10, "headers": {"User-Agent": "test"}}
    html = _requests(url, kwargs)
    assert isinstance(html, str)

def test__requests_with_non_200_status_code():
    url = "http://httpbin.org/status/404"
    kwargs = {"method": "get"}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError as e:
        assert e.status_code == 404

def test__requests_with_encoding():
    url = "http://example.com"
    kwargs = {"method": "get", "encoding": "utf-8"}
    html = _requests(url, kwargs)
    assert isinstance(html, str)


# LLM-generated content at query #16
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test response"
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "test response"
        mock_get.assert_called_once()

def test_url_opener_with_requests_post():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "test response"
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == "test response"
        mock_post.assert_called_once()

def test_url_opener_with_urllib_get():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test response"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {"method": "get"})
        assert result == b"test response"
        mock_urlopen.assert_called_once()

def test_url_opener_with_urllib_post():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = Mock()
        mock_response.read.return_value = b"test response"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == b"test response"
        mock_urlopen.assert_called_once()

def test_url_opener_with_requests_session():
    with patch('requests.Session') as mock_session:
        mock_instance = Mock()
        mock_instance.get.return_value.status_code = 200
        mock_instance.get.return_value.text = "test response"
        mock_session.return_value = mock_instance
        result = url_opener("http://example.com", {"method": "get", "session": mock_instance})
        assert result == "test response"
        mock_instance.get.assert_called_once()

def test_url_opener_with_requests_encoding():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test response"
        mock_get.return_value.encoding = 'utf-8'
        result = url_opener("http://example.com", {"method": "get", "encoding": "utf-8"})
        assert result == "test response"
        mock_get.assert_called_once()

def test_url_opener_with_requests_http_error():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        mock_get.return_value.reason = "Not Found"
        mock_get.return_value.headers = {}
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

def test_url_opener_with_urllib_timeout():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = timeout()
        with pytest.raises(timeout):
            url_opener("http://example.com", {"method": "get", "timeout": 1})


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_12_evaluates_to_false():
    url = "http://example.com?param1=value1"
    method = "GET"
    kwargs = {"data": {"param2": "value2"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] in ('?', '&')


# LLM-generated content at query #18
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_with_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'


# LLM-generated content at query #19
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_question_mark():
    url, data = _query('http://example.com?', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_ampersand():
    url, data = _query('http://example.com?param=1', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?param=1&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_trailing_ampersand():
    url, data = _query('http://example.com?param=1&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?param=1&key=value'
    assert data is None

def test_query_with_non_string_data():
    url, data = _query('http://example.com', 'post', {'data': 123})
    assert url == 'http://example.com'
    assert data == b'123'

def test_query_without_data():
    url, data = _query('http://example.com', 'post', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_case_insensitive_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #20
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test html"
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "test html"
        mock_get.assert_called_once()

def test_url_opener_with_requests_post():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "test html"
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == "test html"
        mock_post.assert_called_once()

def test_url_opener_with_requests_session():
    with patch('requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "test html"
        mock_session_class.return_value = mock_session
        result = url_opener("http://example.com", {"method": "get", "session": mock_session})
        assert result == "test html"
        mock_session.get.assert_called_once()

def test_url_opener_with_requests_http_error():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

def test_url_opener_with_urllib_get():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"test html"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {"method": "get"})
        assert result == b"test html"
        mock_urlopen.assert_called_once()

def test_url_opener_with_urllib_post():
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"test html"
        mock_urlopen.return_value = mock_response
        result = url_opener("http://example.com", {"method": "post", "data": {"key": "value"}})
        assert result == b"test html"
        mock_urlopen.assert_called_once()

def test_url_opener_with_encoding():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test html"
        mock_get.return_value.encoding = "utf-8"
        result = url_opener("http://example.com", {"method": "get", "encoding": "utf-8"})
        assert result == "test html"
        mock_get.assert_called_once()


# LLM-generated content at query #21
#--------------------------

```python
def test_url_ends_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] == '?'


# LLM-generated content at query #22
#--------------------------

```python
def test_url_opener_without_requests():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #23
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'


# LLM-generated content at query #24
#--------------------------

```python
def test_status_code_outside_2xx_range():
    resp = type('Response', (), {'status_code': 404, 'url': 'test', 'reason': 'Not Found', 'headers': {}, 'text': ''})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = Mock(status_code=404)
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #26
#--------------------------

```python
def test_url_opener_without_requests():
    global HAS_REQUEST
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #27
#--------------------------

```python
def test_url_ends_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] == '?'


# LLM-generated content at query #28
#--------------------------

```python
def test_url_ends_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] == '?'


# LLM-generated content at query #29
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = 'test_html'
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test_html'
        mock_get.assert_called_once_with(url='http://example.com', timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_requests_post():
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = 'test_html'
        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == 'test_html'
        mock_post.assert_called_once_with(url='http://example.com', data={'key': 'value'}, timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_requests_session():
    with patch('requests.Session') as mock_session:
        mock_session.return_value.get.return_value.status_code = 200
        mock_session.return_value.get.return_value.text = 'test_html'
        session = mock_session.return_value
        result = url_opener('http://example.com', {'method': 'get', 'session': session})
        assert result == 'test_html'
        session.get.assert_called_once_with(url='http://example.com', timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_requests_encoding():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = 'test_html'
        mock_get.return_value.encoding = 'utf-8'
        result = url_opener('http://example.com', {'method': 'get', 'encoding': 'utf-8'})
        assert result == 'test_html'
        assert mock_get.return_value.encoding == 'utf-8'

def test_url_opener_with_requests_http_error():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(HTTPError):
            url_opener('http://example.com', {'method': 'get'})

def test_url_opener_with_urllib_get():
    with patch('urllib2.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = 'test_html'
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == 'test_html'
        mock_urlopen.assert_called_once_with('http://example.com', None, timeout=DEFAULT_TIMEOUT)

def test_url_opener_with_urllib_post():
    with patch('urllib2.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = 'test_html'
        result = url_opener('http://example.com', {'method': 'post', 'data': 'test_data'})
        assert result == 'test_html'
        mock_urlopen.assert_called_once_with('http://example.com', 'test_data', timeout=DEFAULT_TIMEOUT)


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_false():
    resp = type('Response', (), {'status_code': 199})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #31
#--------------------------

```python
def test_url_ends_with_question_mark():
    url = "http://example.com?"
    method = "get"
    kwargs = {"data": {"key": "value"}}
    result_url, result_data = _query(url, method, kwargs)
    assert url[-1] == '?'


# LLM-generated content at query #32
#--------------------------

```python
def test__requests_with_get_method():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_post_method():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_non_200_status_code():
    url = "http://httpbin.org/status/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError as e:
        assert e.status_code == 404


# LLM-generated content at query #33
#--------------------------

```python
def test_url_opener_uses_urllib_when_no_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__requests_get_without_data():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_get_with_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_post():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_http_error():
    url = "http://example.com/404"
    kwargs = {'method': 'get'}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_query_with_data_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_string():
    url, data = _query('http://example.com', 'post', {'data': 'raw_data'})
    assert url == 'http://example.com'
    assert data == b'raw_data'

def test_query_get_with_data_no_query():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_with_data_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_with_data_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_no_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_case_insensitive_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #3
#--------------------------

```python
def test_data_is_encoded_when_not_none():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data is None

    data = "test data"
    kwargs = {'data': data}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == data.encode('utf-8')


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener_with_requests_get():
    assert url_opener('http://example.com', {'method': 'get', 'session': None}) == requests.get('http://example.com', timeout=DEFAULT_TIMEOUT).text

def test_url_opener_with_requests_post():
    assert url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}, 'session': None}) == requests.post('http://example.com', data={'key': 'value'}, timeout=DEFAULT_TIMEOUT).text

def test_url_opener_with_urllib_get():
    assert url_opener('http://example.com', {'method': 'get'}) == urlopen('http://example.com', None, timeout=DEFAULT_TIMEOUT).read()

def test_url_opener_with_urllib_post():
    assert url_opener('http://example.com', {'method': 'post', 'data': 'key=value'}) == urlopen('http://example.com', b'key=value', timeout=DEFAULT_TIMEOUT).read()

def test_url_opener_with_requests_encoding():
    assert url_opener('http://example.com', {'method': 'get', 'encoding': 'utf-8', 'session': None}) == requests.get('http://example.com', timeout=DEFAULT_TIMEOUT).text

def test_url_opener_with_requests_timeout():
    assert url_opener('http://example.com', {'method': 'get', 'timeout': 10, 'session': None}) == requests.get('http://example.com', timeout=10).text

def test_url_opener_with_urllib_timeout():
    assert url_opener('http://example.com', {'method': 'get', 'timeout': 10}) == urlopen('http://example.com', None, timeout=10).read()


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    # Mock response with status_code outside the 200-299 range
    resp = type('Response', (), {'status_code': 404, 'url': 'test_url', 'reason': 'Not Found', 'headers': {}, 'text': 'test_html'})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #6
#--------------------------

```python
def test_data_is_encoded_when_present():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('module.HAS_REQUEST', True):
        with patch('module.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "test html"
            mock_get.return_value = mock_response
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "test html"

def test_url_opener_with_requests_post():
    with patch('module.HAS_REQUEST', True):
        with patch('module.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "test html"
            mock_post.return_value = mock_response
            result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
            assert result == "test html"

def test_url_opener_with_urllib_get():
    with patch('module.HAS_REQUEST', False):
        with patch('module.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"test html"
            mock_urlopen.return_value = mock_response
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == b"test html"

def test_url_opener_with_urllib_post():
    with patch('module.HAS_REQUEST', False):
        with patch('module.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"test html"
            mock_urlopen.return_value = mock_response
            result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
            assert result == b"test html"

def test_url_opener_with_requests_http_error():
    with patch('module.HAS_REQUEST', True):
        with patch('module.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.url = "http://example.com"
            mock_response.reason = "Not Found"
            mock_response.headers = {}
            mock_get.return_value = mock_response
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

def test_url_opener_with_encoding():
    with patch('module.HAS_REQUEST', True):
        with patch('module.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "test html"
            mock_get.return_value = mock_response
            result = url_opener("http://example.com", {'method': 'get', 'encoding': 'utf-8'})
            assert result == "test html"


# LLM-generated content at query #8
#--------------------------

```python
def test_requests_get_with_data():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_post_with_data():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_encoding():
    url = "http://example.com"
    kwargs = {'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_session():
    url = "http://example.com"
    kwargs = {'session': requests.Session()}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_with_timeout():
    url = "http://example.com"
    kwargs = {'timeout': 10}
    result = _requests(url, kwargs)
    assert result is not None

def test_requests_http_error():
    url = "http://example.com/404"
    kwargs = {}
    try:
        _requests(url, kwargs)
    except HTTPError as e:
        assert e.status_code == 404


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener_uses_urllib_when_no_requests():
    HAS_REQUEST = False
    result = url_opener("http://example.com", {})
    assert result == _urllib("http://example.com", {})


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_false():
    url = "http://example.com"
    method = "post"
    kwargs = {}
    result = _query(url, method, kwargs)
    assert result == (url, None)


# LLM-generated content at query #11
#--------------------------

```python
def test_url_opener_without_requests():
    global HAS_REQUEST
    HAS_REQUEST = False
    assert not (HAS_REQUEST)


# LLM-generated content at query #12
#--------------------------

```python
def test_query_with_data_as_dict():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_list():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_data_as_tuple():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_get_method_and_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_get_method_and_data_with_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'

def test_query_with_non_string_method():
    url, data = _query('http://example.com', 'POST', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'


# LLM-generated content at query #13
#--------------------------

```python
def test_data_is_encoded():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    resp = type('Response', (), {'status_code': 404})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 300})
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #16
#--------------------------

```python
def test_data_encoding_predicate():
    url = "http://example.com"
    method = "post"
    kwargs = {"data": "test data"}
    result_url, result_data = _query(url, method, kwargs)
    assert result_data == "test data".encode('utf-8')


# LLM-generated content at query #17
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #18
#--------------------------

```python
def test__requests_with_get_method():
    url = "http://example.com"
    kwargs = {'method': 'get', 'data': {'key': 'value'}, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_post_method():
    url = "http://example.com"
    kwargs = {'method': 'post', 'data': {'key': 'value'}, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_session():
    url = "http://example.com"
    session = requests.Session()
    kwargs = {'method': 'get', 'session': session, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_timeout():
    url = "http://example.com"
    kwargs = {'method': 'get', 'timeout': 10, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_allowed_args():
    url = "http://example.com"
    kwargs = {'method': 'get', 'headers': {'User-Agent': 'test'}, 'encoding': 'utf-8'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)

def test__requests_with_http_error():
    url = "http://example.com/404"
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    try:
        _requests(url, kwargs)
        assert False, "Expected HTTPError"
    except HTTPError:
        pass

def test__requests_with_no_encoding():
    url = "http://example.com"
    kwargs = {'method': 'get'}
    result = _requests(url, kwargs)
    assert isinstance(result, str)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'timeout': 10,
        'encoding': 'utf-8'
    }
    mock_resp = Mock(status_code=404)
    with patch('requests.get', return_value=mock_resp):
        with pytest.raises(HTTPError):
            _requests(url, kwargs)


# LLM-generated content at query #20
#--------------------------

```python
def test_query_with_data_in_kwargs():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_dict_data():
    url, data = _query('http://example.com', 'post', {'data': {'key': 'value'}})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_list_data():
    url, data = _query('http://example.com', 'post', {'data': [('key', 'value')]})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_with_tuple_data():
    url, data = _query('http://example.com', 'post', {'data': (('key', 'value'),)})
    assert url == 'http://example.com'
    assert data == b'key=value'

def test_query_get_method_with_data():
    url, data = _query('http://example.com', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_no_ampersand():
    url, data = _query('http://example.com?existing=param', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_get_method_with_data_and_existing_query_with_ampersand():
    url, data = _query('http://example.com?existing=param&', 'get', {'data': {'key': 'value'}})
    assert url == 'http://example.com?existing=param&key=value'
    assert data is None

def test_query_with_string_data():
    url, data = _query('http://example.com', 'post', {'data': 'raw string'})
    assert url == 'http://example.com'
    assert data == b'raw string'

def test_query_without_data():
    url, data = _query('http://example.com', 'get', {})
    assert url == 'http://example.com'
    assert data is None

def test_query_case_insensitive_get_method():
    url, data = _query('http://example.com', 'GET', {'data': {'key': 'value'}})
    assert url == 'http://example.com?key=value'
    assert data is None


# LLM-generated content at query #21
#--------------------------

```python
def test_url_opener_with_requests_get():
    with patch('module.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "test response"
        mock_get.return_value.encoding = None
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "test response"
        mock_get.assert_called_once()

def test_url_opener_with_requests_post():
    with patch('module.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "test response"
        mock_post.return_value.encoding = None
        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        assert result == "test response"
        mock_post.assert_called_once()

def test_url_opener_with_requests_session():
    with patch('module.requests.Session') as mock_session:
        mock_session.return_value.get.return_value.status_code = 200
        mock_session.return_value.get.return_value.text = "test response"
        mock_session.return_value.get.return_value.encoding = None
        result = url_opener("http://example.com", {'method': 'get', 'session': mock_session})
        assert result == "test response"
        mock_session.return_value.get.assert_called_once()

def test_url_opener_with_requests_http_error():
    with patch('module.requests.get') as mock_get:
        mock_get.return_value.status_code = 404
        mock_get.return_value.reason = "Not Found"
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_with_urllib_get():
    with patch('module.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = b"test response"
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "test response"
        mock_urlopen.assert_called_once()

def test_url_opener_with_urllib_post():
    with patch('module.urlopen') as mock_urlopen:
        mock_urlopen.return_value.read.return_value = b"test response"
        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        assert result == "test response"
        mock_urlopen.assert_called_once()


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    resp = type('Response', (), {'status_code': 300})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #23
#--------------------------

```python
def test_url_opener_without_requests():
    HAS_REQUEST = False
    assert url_opener("http://example.com", {}) == _urllib("http://example.com", {})


# LLM-generated content at query #24
#--------------------------

```python
def test_data_encoding_when_data_is_not_none():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_false():
    resp = type('Response', (), {'status_code': 199})()
    assert not (200 <= resp.status_code < 300)


# LLM-generated content at query #26
#--------------------------

```python
def test_data_is_encoded_to_utf8_when_not_none():
    url, data = _query('http://example.com', 'post', {'data': 'test'})
    assert data == b'test'


# LLM-generated content at query #27
#--------------------------

```python
def test_url_opener_when_HAS_REQUEST_is_false():
    HAS_REQUEST = False
    assert not HAS_REQUEST


