####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("has_requests", [True, False])
def test_url_opener(has_requests):
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'timeout': 10,
        'data': {'key': 'value'},
        'headers': {'User-Agent': 'test'}
    }

    if has_requests:
        with patch('__main__.HAS_REQUEST', True), \
             patch('requests.get') as mock_get:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>success</html>"
            mock_response.url = url
            mock_response.reason = "OK"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            result = url_opener(url, kwargs)

            assert result == "<html>success</html>"
            # Check if _query correctly appended data to URL
            expected_url = "http://example.com?key=value"
            args, kwargs_passed = mock_get.call_args
            assert kwargs_passed['url'] == expected_url
            assert kwargs_passed['timeout'] == 10
            assert kwargs_passed['headers'] == {'User-Agent': 'test'}

            # Test HTTP Error
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            with pytest.raises(HTTPError):
                url_opener(url, kwargs)

    else:
        with patch('__main__.HAS_REQUEST', False), \
             patch('__main__.urlopen') as mock_urlopen:
            
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            result = url_opener(url, kwargs)

            expected_url = "http://example.com?key=value"
            # urllib.request.urlopen handles data as bytes if encoded in _query
            args, kwargs_passed = mock_urlopen.call_args
            assert args[0] == expected_url
            assert args[1] == b'key=value'
            assert kwargs_passed['timeout'] == 10
            assert result == mock_response

@pytest.mark.parametrize("method, data_input, expected_url", [
    ('get', {'a': 'b'}, "http://test.com?a=b"),
    ('get', None, "http://test.com"),
    ('post', {'a': 'b'}, "http://test.com"), # Data stays in body for POST
    ('get', 'extra_params', "http://test.com?extra_params"),
])
def test_query_logic(method, data_input, expected_url):
    url = "http://test.com"
    kwargs = {'method': method}
    if data_input:
        kwargs['data'] = data_input
    
    # Test _query directly to validate URL construction logic
    res_url, res_data = _query(url, method, kwargs)
    
    if method.lower() == 'get' and data_input:
        assert res_url == expected_url
    else:
        assert res_url == url

def test_requests_session_usage():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_data"
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://test.com", kwargs)
        
        assert result == "session_data"
        mock_session.get.assert_called_once()

def test_query_url_with_existing_params():
    url = "http://test.com?existing=1"
    kwargs = {'method': 'get', 'data': {'new': '2'}}
    res_url, _ = _query(url, 'get', kwargs)
    assert res_url == "http://test.com?existing=1&new=2"

def test_query_url_with_trailing_separator():
    url = "http://test.com?"
    kwargs = {'method': 'get', 'data': {'new': '2'}}
    res_url, _ = _query(url, 'get', kwargs)
    assert res_url == "http://test.com?new=2"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>Success</html>"
            mock_response.url = "http://example.com"
            mock_get.return_value = mock_response

            kwargs = {'timeout': 10, 'data': {'key': 'val'}}
            result = url_opener("http://example.com", kwargs)

            assert result == "<html>Success</html>"
            # Check if data was encoded into query params in URL
            args, kwargs_call = mock_get.call_args
            assert "http://example.com?key=val" in args[0] or "http://example.com?key%3Dval" in args[0]
            assert kwargs_call['timeout'] == 10

    # Test case 2: Using requests with HTTP Error (404)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            mock_response.url = "http://example.com/bad"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener("http://example.com/bad", {})

    # Test case 3: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            kwargs = {'method': 'GET'}
            url_opener("http://example.com", kwargs)
            
            mock_urlopen.assert_called_once()

    # Test case 4: _query logic - handling existing query parameters
    # Check if it appends correctly with '&'
    kwargs = {'data': {'a': '1'}, 'method': 'GET'}
    url, data = _query("http://example.com?existing=true", "get", kwargs)
    assert "existing=true" in url
    assert "a=1" in url
    assert "&" in url or "?" in url

    # Test case 5: _query logic - handling POST-like data (not GET)
    kwargs = {'data': {'key': 'value'}, 'method': 'POST'}
    url, data = _query("http://example.com", "post", kwargs)
    assert url == "http://example.com"
    assert data == b'key=value'

    # Test case 6: _requests with session object
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Session Content"
        mock_session.get.return_value = mock_response
        
        kwargs = {'session': mock_session, 'method': 'get'}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "Session Content"
        mock_session.get.assert_called_once()

    # Test case 7: _requests with encoding
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "utf-8 content"
            mock_get.return_value = mock_response
            
            kwargs = {'encoding': 'utf-8'}
            url_opener("http://example.com", kwargs)
            assert mock_response.encoding == 'utf-8'
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test GET with params via _query logic (data converted to query string)
            kwargs = {'data': {'key': 'value'}, 'timeout': 10}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            # Check if data was appended to URL via _query
            called_url = mock_get.call_args[1]['url']
            assert "key=value" in called_url
            assert mock_get.call_args[1]['timeout'] == 10

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Test POST with data (data passed as bytes)
            kwargs = {'data': {'key': 'value'}, 'method': 'post'}
            url_opener("http://example.com", kwargs)
            
            called_data = mock_post.call_args[1]['data']
            assert called_data == b'key=value'

        # Test HTTP Error raising
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.url = "http://example.com"
        mock_error_resp.reason = "Not Found"
        mock_error_resp.headers = {}
        
        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 2: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urllib_resp = MagicMock()
            mock_urllib_resp.read.return_value = b"urllib content"
            mock_urlopen.return_value = mock_urllib_resp
            
            # We need to patch the return of urlopen to simulate what it returns 
            # (Note: _urllib returns the object from urlopen, not .read())
            # Since urlopen returns a file-like object, we just ensure it's called.
            
            kwargs = {'method': 'GET', 'timeout': 5}
            url_opener("http://example.com", kwargs)
            
            mock_urlopen.assert_called_once()
            args, kwargs_call = mock_urlopen.call_args
            assert args[0] == "http://example.com"
            assert kwargs_call['timeout'] == 5

    # Test Case 3: Testing _query edge cases (URL formatting)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = ""
            
            # Test URL with existing query params
            url_opener("http://example.com?a=1", {'data': {'b': '2'}, 'method': 'get'})
            called_url = mock_get.call_args[1]['url']
            assert "a=1" in called_url
            assert "b=2" in called_url
            assert "&" in called_url or "?" in called_url

            # Test URL with trailing slash/char logic
            url_opener("http://example.com/?", {'data': {'b': '2'}, 'method': 'get'})
            called_url = mock_get.call_args[1]['url']
            assert "b=2" in called_url
            assert not called_url.endswith('?')
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url_suffix", [
    ('get', {'data': {'key': 'val'}}, '?key=val'),
    ('get', {'data': [{'a': 1}]}, '?a=1'),
    ('get', {'data': 'raw_string'}, '?raw_string'),
    ('post', {'data': {'key': 'val'}}, ''), # POST doesn't append to URL in _query logic
])
def test_url_opener_logic(method, params, expected_url_suffix):
    kwargs = {'method': method}
    kwargs.update(params)
    
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_get.return_value = mock_response
        
        url = "http://example.com"
        result = url_opener(url, kwargs)
        
        # Verify URL construction
        called_url = mock_get.call_args[1]['url']
        if expected_url_suffix:
            assert called_url.endswith(expected_url_suffix)
        else:
            assert called_url == url
        
        assert result == "success"

def test_url_opener_http_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://example.com/bad"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com/bad", {'method': 'get'})

def test_url_opener_urllib_fallback():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_urlopen.return_value = MagicMock(read=lambda: b"html_content")
        # Note: urlopen returns a file-like object, we simulate the return of _urllib
        # Since _urllib returns what urlopen returns, and urlopen usually returns bytes/stream
        
        url = "http://example.com"
        kwargs = {'method': 'get', 'data': {'a': 1}}
        
        url_opener(url, kwargs)
        
        args, kwargs_call = mock_urlopen.call_args
        assert args[0] == "http://example.com?a=1"
        assert kwargs_call['timeout'] == 60

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_success"
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        
        mock_session.get.assert_called_once()
        assert result == "session_success"

def test_url_opener_allowed_args_passing():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "ok"
        
        kwargs = {
            'method': 'post',
            'auth': ('user', 'pass'),
            'headers': {'X-Test': 'true'},
            'proxies': {'http': 'proxy_url'},
            'extra_unallowed_arg': 'ignore_me'
        }
        
        url_opener("http://example.com", kwargs)
        
        passed_kwargs = mock_get.call_args[1]
        assert passed_kwargs['auth'] == ('user', 'pass')
        assert passed_kwargs['headers'] == {'X-Test': 'true'}
        assert passed_kwargs['proxies'] == {'http': 'proxy_url'}
        assert 'extra_unallowed_arg' not in passed_kwargs

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "utf8_content"
        mock_get.return_value = mock_response
        
        kwargs = {'method': 'get', 'encoding': 'utf-8'}
        result = url_opener("http://example.com", kwargs)
        
        assert mock_response.encoding == 'utf-8'
        assert result == "utf8_content"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method", ["get", "post"])
def test_url_opener(method):
    # Setup mocks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>success</html>"
    mock_response.url = "http://example.com"
    mock_response.reason = "OK"
    mock_response.headers = {}

    # Case 1: Test using requests if available
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get' if method == 'get' else 'requests.post', return_value=mock_response) as mock_meth:
            kwargs = {'timeout': 10, 'data': {'key': 'val'}}
            # For GET, _query converts data to query string
            if method == 'get':
                expected_url = "http://example.com?key=val"
            else:
                expected_url = "http://example.com"
                
            result = url_opener(expected_url, kwargs)
            
            assert result == "<html>success</html>"
            mock_meth.assert_called_once()

    # Case 2: Test using urllib if requests is not available
    with patch('__main__.HAS_REQUEST', False):
        with patch('__main__.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock(read=lambda: b"html_content")
            # We need to simulate the behavior of urlopen returning something with .read() 
            # but since _urllib returns the object, we check if it calls urlopen correctly
            kwargs = {'method': method, 'data': {'a': 'b'}}
            url_opener("http://example.com", kwargs)
            
            args, kwargs_call = mock_urlopen.call_args
            assert "a=b" in args[0]

    # Case 3: Test HTTP Error raising
    with patch('__main__.HAS_REQUEST', True):
        mock_error_resp = MagicMock()
        mock_error_resp.status_code = 404
        mock_error_resp.url = "http://example.com"
        mock_error_resp.reason = "Not Found"
        mock_error_resp.headers = {}
        
        with patch('requests.get', return_value=mock_error_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Case 4: Test session usage in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        with patch('requests.get') as mock_req_get:
            kwargs = {'method': 'get', 'session': mock_session}
            url_opener("http://example.com", kwargs)
            mock_session.get.assert_called()
            mock_req_get.assert_not_called()

    # Case 5: Test encoding parameter
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get', return_value=mock_response):
            kwargs = {'method': 'get', 'encoding': 'utf-8'}
            result = url_opener("http://example.com", kwargs)
            assert mock_response.encoding == 'utf-8'
            assert result == "<html>success</html>"

def test_query_logic_edge_cases():
    # Test URL construction with existing params
    kwargs = {'data': {'id': '123'}}
    url, data = _query("http://test.com?existing=true", "get", kwargs)
    assert url == "http://test.com?existing=true&id=123"

    # Test URL construction with trailing separator
    kwargs = {'data': {'id': '123'}}
    url, data = _query("http://test.com?", "get", kwargs)
    assert url == "http://test.com?id=123"

    # Test POST data encoding to bytes
    kwargs = {'data': {'key': 'value'}}
    _, data_bytes = _query("http://test.com", "post", kwargs)
    assert isinstance(data_bytes, bytes)
    assert b"key=value" in data_bytes
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>success</html>"
        mock_resp.url = "http://example.com"
        mock_resp.reason = "OK"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp) as mock_get:
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "<html>success</html>"
            mock_get.assert_called_once()

    # Test case 2: Using requests with data/params transformation (GET method)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "data_encoded"
        
        with patch('requests.get', return_value=mock_resp) as mock_get:
            kwargs = {'method': 'get', 'data': {'key': 'val'}}
            url_opener("http://example.com", kwargs)
            # Check if url was appended with encoded data
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']

    # Test case 3: Using requests error handling (HTTPError)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.url = "http://example.com"
        mock_resp.reason = "Not Found"
        mock_resp.headers = {}

        with patch('requests.get', return_value=mock_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_response = MagicMock()
        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            # Simulate read() behavior for urlopen response
            mock_response.read.return_value = b"urllib_content"
            
            # Note: _urllib returns the urlopen object, not text directly
            res = url_opener("http://example.com", {'method': 'get'})
            assert res == mock_response
            mock_urlopen.assert_called_once()

    # Test case 5: Using requests with session object
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_content"
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        assert result == "session_content"
        mock_session.get.assert_called_once()

    # Test case 6: Verifying allowed_args pass through
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "headers_test"
        
        kwargs = {'method': 'get', 'headers': {'User-Agent': 'Test'}, 'timeout': 10}
        with patch('requests.get', return_value=mock_resp) as mock_get:
            url_opener("http://example.com", kwargs)
            _, call_kwargs = mock_get.call_args
            assert call_kwargs['headers'] == {'User-Agent': 'Test'}
            assert call_kwargs['timeout'] == 10
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Mocking dependencies and environment
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Setup mock response for successful GET
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_get.return_value = mock_response

        # Test Case 1: Successful GET with params via requests
        kwargs = {'params': {'key': 'val'}, 'timeout': 10}
        # Note: _requests implementation uses 'data' from kwargs for query construction
        result = url_opener("http://example.com", {'data': {'a': 1}, 'timeout': 5})
        assert result == "<html>Success</html>"
        mock_get.assert_called()

        # Test Case 2: HTTP Error handling in _requests
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        mock_error_response.reason = "Not Found"
        mock_error_response.url = "http://example.com/404"
        mock_error_response.headers = {}
        mock_get.return_value = mock_error_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com/404", {})

        # Test Case 3: POST request via requests
        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post_response.text = "Created"
        mock_post.return_value = mock_post_response
        
        result_post = url_opener("http://example.com/post", {'method': 'post', 'data': 'payload'})
        assert result_post == "Created"

    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        # Setup mock for urllib
        mock_urlopen.return_value = MagicMock(read=lambda: b"urllib_data")
        
        # Test Case 4: urllib fallback (GET)
        # Note: urlopen returns a file-like object, so we check the call signature
        url_opener("http://example.com/urllib", {'method': 'get'})
        mock_urlopen.assert_called()

        # Test Case 5: urllib with data (POST-like)
        url_opener("http://example.com/data", {'method': 'post', 'data': {'id': '123'}})
        args, kwargs = mock_urlopen.call_args
        assert "id=123" in args[0]

    # Test Case 6: _query logic for URL encoding and appending
    from __main__ import _query
    
    # No existing query params
    url, data = _query("http://test.com", "get", {'data': {'a': 1}})
    assert url == "http://test.com?a=1"
    assert data is None

    # Existing query params (append with &)
    url, data = _query("http://test.com?existing=true", "get", {'data': {'b': 2}})
    assert url == "http://test.com?existing=true&b=2"

    # Method is not GET (do not append to URL)
    url, data = _query("http://test.com", "post", {'data': {'a': 1}})
    assert url == "http://test.com"
    assert data == b'a=1'
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize("method, params, expected_url", [
    ("GET", {"data": {"key": "val"}}, "http://test.com?key=val"),
    ("GET", {"data": [("a", "1"), ("b", "2")]}, "http://test.com?a=1&b=2"),
    ("GET", {"data": {"key": "val"}}, "http://test.com/path?key=val"),
])
def test_query_logic(method, params, expected_url):
    # Test the underlying _query logic used by both engines
    kwargs = {'data': params['data']} if 'data' in params else {}
    url, data = _query("http://test.com/path", method, kwargs)
    assert url == expected_url

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_response.url = "http://test.com"
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {"method": "get"})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://test.com"
        mock_response.headers = {}
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://test.com", {"method": "get"})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_response.read.return_value = b"html_content"
        # urllib.request.urlopen returns a file-like object, 
        # but the code uses it directly in _urllib return.
        # We mock the return to behave like an opened stream.
        mock_urlopen.return_value = MagicMock(read=lambda: b"html_content")

        url_opener("http://test.com", {"method": "get"})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_allowed_args():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response

        kwargs = {
            'method': 'get',
            'headers': {'User-Agent': 'test'},
            'cookies': {'session': '123'},
            'timeout': 10
        }
        
        url_opener("http://test.com", kwargs)
        
        # Verify only allowed args are passed to the request call
        args, kwargs_passed = mock_get.call_args
        assert 'headers' in kwargs_passed
        assert kwargs_passed['headers'] == {'User-Agent': 'test'}
        assert 'cookies' in kwargs_passed
        assert 'timeout' in kwargs_passed
        # Ensure 'method' (not in allowed_args) was handled by _query/logic, not passed as kwarg
        assert 'method' not in kwargs_passed

def test_url_opener_get_with_existing_params():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response

        # Test that it appends '&' if '?' is already present
        url_opener("http://test.com?existing=true", {"data": {"new": "val"}})
        
        args, kwargs_passed = mock_get.call_args
        assert kwargs_passed['url'] == "http://test.com?existing=true&new=val"

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "utf8_content"
        mock_get.return_value = mock_response

        url_opener("http://test.com", {"encoding": "utf-8"})
        
        assert mock_response.encoding == "utf-8"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test basic GET
            result = url_opener("http://example.com", {"method": "get"})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

            # Test GET with data/params encoding
            kwargs = {"method": "get", "data": {"key": "val"}}
            url_opener("http://example.com", kwargs)
            # Check if url was modified to include query string
            args, kwargs_call = mock_get.call_args
            assert "key=val" in args[0]

            # Test POST with data
            mock_post_response = MagicMock()
            mock_post_response.status_code = 201
            mock_post_response.text = "Created"
            with patch('requests.post', return_value=mock_post_response) as mock_post:
                url_opener("http://example.com", {"method": "post", "data": "body"})
                # Verify data is encoded to bytes in _query logic if it passes through
                # Note: _query handles dict/list/tuple for GET, but POST uses raw data
                pass

            # Test HTTP Error (Non 2xx)
            mock_error_resp = MagicMock()
            mock_error_resp.status_code = 404
            mock_error_resp.reason = "Not Found"
            mock_error_resp.url = "http://example.com"
            mock_error_resp.headers = {}
            with patch('requests.get', return_value=mock_error_resp):
                with pytest.raises(HTTPError):
                    url_opener("http://example.com", {"method": "get"})

    # Test Case 2: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_stream = MagicMock()
            mock_stream.read.return_value = b"urllib content"
            # Note: urlopen returns a response object that behaves like a file
            # In the code, it returns the result of urlopen directly
            mock_urlopen.return_value = mock_stream
            
            # We need to mock the return value behavior for _urllib's use case
            # Since _urllib returns the result of urlopen, and we expect a stream/response
            # The test checks if urlopen is called with correct params
            url_opener("http://example.com", {"method": "get"})
            mock_urlopen.assert_called()

    # Test Case 3: _query logic specifically (Edge cases)
    # Testing URL concatenation for GET with existing query params
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = ""
            
            # Case: URL already has '?'
            url_opener("http://example.com?a=1", {"method": "get", "data": {"b": "2"}})
            args, _ = mock_get.call_args
            assert "a=1&b=2" in args[0]

            # Case: URL has '?' but ends with '&'
            url_opener("http://example.com?", {"method": "get", "data": {"b": "2"}})
            args, _ = mock_get.call_args
            assert "b=2" in args[0]
            assert not args[0].endswith('&')

    # Test Case 4: session usage
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session content"
        
        url_opener("http://example.com", {"method": "get", "session": mock_session})
        mock_session.get.assert_called_once()
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Requests is available (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html>Success</html>"
            mock_response.url = "http://example.com"
            mock_response.reason = "OK"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
            result = url_opener("http://example.com", kwargs)

            assert result == "<html>Success</html>"
            mock_get.assert_called_once()
            # Verify allowed_args filtering (headers should be passed)
            args, call_kwargs = mock_get.call_args
            assert call_kwargs['headers'] == {'User-Agent': 'test'}
            assert call_kwargs['timeout'] == 10

    # Test Case 2: Requests is available, but returns Error (404)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.url = "http://example.com"
            mock_response.reason = "Not Found"
            mock_response.headers = {}
            mock_get.return_value = mock_response

            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 3: Requests is NOT available (HAS_REQUEST = False) - Using urllib
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            url = "http://example.com"
            kwargs = {'method': 'get', 'data': {'key': 'value'}}
            
            url_opener(url, kwargs)
            
            # Check if _query correctly appended data to URL for GET
            expected_url = "http://example.com?key=value"
            mock_urlopen.assert_called_once()
            args, _ = mock_urlopen.call_args
            assert args[0] == expected_url

    # Test Case 4: Testing _query logic for POST-like data (data remains in body)
    kwargs_post = {'method': 'post', 'data': {'name': 'test'}}
    url, data = _query("http://example.com", "post", kwargs_post)
    assert url == "http://example.com"
    # Data should be urlencoded and bytes in the return
    assert b'name=test' in data or data == b'name%3Dtest' # depends on implementation specifics

    # Test Case 5: Testing _query logic for GET with existing query params
    kwargs_get = {'method': 'get', 'data': {'a': 'b'}}
    url, data = _query("http://example.com?existing=1", "get", kwargs_get)
    assert "existing=1" in url
    assert "a=b" in url
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test GET with params in kwargs (should be moved to URL by _query)
            kwargs = {'data': {'key': 'val'}, 'timeout': 10}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            # Check if _query appended the data to the URL
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']
            assert kwargs_call['timeout'] == 10

        # Test case 2: requests raising HTTPError (non-2xx status)
        with patch('requests.get') as mock_get_error:
            mock_err_resp = MagicMock()
            mock_err_resp.status_code = 404
            mock_err_resp.url = "http://example.com"
            mock_err_resp.reason = "Not Found"
            mock_err_resp.headers = {}
            mock_get_error.return_value = mock_err_resp
            
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test case 3: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urllib_resp = MagicMock()
        # urllib.request.urlopen returns a file-like object; we mock the read behavior if needed, 
        # but here we just care about it returning the object itself.
        with patch('urllib.request.urlopen', return_value=mock_urllib_resp) as mock_urlopen:
            kwargs = {'method': 'get', 'data': {'a': 'b'}}
            url_opener("http://example.com", kwargs)
            
            # Verify URL was encoded correctly for urllib
            called_url, called_data, called_timeout = mock_urlopen.call_args[0]
            assert "a=b" in called_url or called_data is not None
            assert called_timeout == 60

    # Test case 4: Verify allowed_args filtering in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # 'invalid_arg' should be filtered out, but 'headers' should stay
            kwargs = {'headers': {'User-Agent': 'test'}, 'invalid_arg': 'ignore_me'}
            url_opener("http://example.com", kwargs)
            
            _, kwargs_call = mock_get.call_args
            assert 'headers' in kwargs_call
            assert kwargs_call['headers'] == {'User-Agent': 'test'}
            assert 'invalid_arg' not in kwargs_call

    # Test case 5: Verify encoding application in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", {'encoding': 'utf-16'})
            assert mock_response.encoding == 'utf-16'
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Successful GET request using requests (if available)
    if HAS_REQUEST:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {"method": "get"})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

    # Test Case 2: Successful GET request using urllib (fallback/standard)
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>Urllib</html>"
        mock_urlopen.return_value = mock_response
        
        # We force the logic to use urllib path by mocking HAS_REQUEST to False
        with patch('__main__.HAS_REQUEST', False):
            result = url_opener("http://example.com", {"method": "get"})
            # Note: urlopen returns a response object, calling .read() manually in real usage
            # but here we test the function's return value which is the response object itself
            assert result == mock_response
            mock_urlopen.assert_called_once()

    # Test Case 3: Testing _query logic via url_opener with params
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            kwargs = {'method': 'get', 'data': {'key': 'value'}}
            url_opener("http://example.com", kwargs)
            # Check if url was encoded correctly
            args, _ = mock_urlopen.call_args
            assert "http://example.com?key=value" in args[0]

    # Test Case 4: HTTP Error handling in requests path
    if HAS_REQUEST:
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        mock_error_response.url = "http://example.com"
        mock_error_response.reason = "Not Found"
        mock_error_response.headers = {}

        with patch('requests.get', return_value=mock_error_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {"method": "get"})

    # Test Case 5: Verifying allowed arguments filtering in requests path
    if HAS_REQUEST:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        
        extra_arg = 'unsupported_arg'
        kwargs = {'method': 'get', extra_arg: 'should_be_filtered'}
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", kwargs)
            # Check that the call to requests.get does NOT contain the unsupported arg
            args, kwargs_passed = mock_get.call_args
            assert extra_arg not in kwargs_passed
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {"method": "get"})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

    # Test case 2: Using requests with query parameters (GET)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "data"
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {"method": "get", "data": {"key": "val"}}
            url_opener("http://example.com", kwargs)
            # Check if url was encoded correctly
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']

    # Test case 3: Using requests with HTTP Error
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response import HTTPError

        with patch('requests.get', return_value=mock_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {"method": "get"})

    # Test case 4: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            url_opener("http://example.com", {"method": "get"})
            mock_urlopen.assert_called_once()

    # Test case 5: Using urllib with data (POST-like)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            kwargs = {"method": "post", "data": {"a": "b"}}
            url_opener("http://example.com", kwargs)
            args, kwargs_call = mock_urlopen.call_args
            # check that data is encoded to bytes in _query logic for urllib
            assert b"a=b" in args[1]

    # Test case 6: Testing allowed_args passing (headers)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        headers = {'User-Agent': 'test'}
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", {"headers": headers})
            _, kwargs_call = mock_get.call_args
            assert kwargs_call['headers'] == headers
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ('get', {'data': {'key': 'val'}}, 'http://test.com?key=val'),
    ('get', {'data': 'raw_string'}, 'http://test.com?raw_string'),
    ('get', {'data': {'a': 1}}, 'http://test.com?a=1'),
    ('post', {'data': {'key': 'val'}}, 'http://test.com'),
])
def test_query_logic(method, params, expected_url):
    from urllib.parse import urlencode
    kwargs = {'data': params.copy()} if 'data' in params else {}
    # Test internal _query logic
    url, data = _query('http://test.com', method, kwargs)
    if method.lower() == 'get' and 'data' in params:
        assert url == expected_url
    if method.lower() == 'post' and 'data' in params:
        # In post mode, data is not appended to URL but returned as bytes
        expected_encoded = urlencode(params['data']).encode('utf-8')
        assert data == expected_encoded

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_response.url = "http://test.com"
        mock_get.return_value = mock_response

        result = url_opener("http://test.com", {'method': 'get', 'timeout': 10})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_failure():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_response.url = "http://test.com"
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            url_opener("http://test.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('__main__.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://test.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()

def test_url_opener_with_session():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.Session') as mock_session_class:
        
        mock_session = mock_session_class.return_value
        mock_method = MagicMock()
        mock_method.status_code = 200
        mock_method.text = "session_data"
        mock_session.get = mock_method
        
        kwargs = {'session': mock_session, 'method': 'get'}
        result = url_opener("http://test.com", kwargs)
        
        assert result == "session_data"
        mock_session.get.assert_called_once()

def test_url_opener_encoding():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_get.return_value = mock_response

        url_opener("http://test.com", {'encoding': 'utf-16'})
        assert mock_response.encoding == 'utf-16'

def test_query_edge_cases():
    # Test URL already has params
    url, data = _query('http://test.com?existing=true', 'get', {'data': {'new': 'val'}})
    assert url == 'http://test.com?existing=true&new=val'

    # Test URL ends with &
    url, data = _query('http://test.com?a=b&', 'get', {'data': {'c': 'd'}})
    assert url == 'http://test.com?a=b&c=d'
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test GET with params in kwargs
            kwargs = {'params': {'key': 'val'}, 'timeout': 10}
            # Note: _query handles 'data' key specifically for encoding
            kwargs_with_data = {'data': {'a': 1}, 'timeout': 5}
            
            result = url_opener("http://example.com", kwargs_with_data)
            
            assert result == "<html>Success</html>"
            # Check if url was modified by _query (data converted to query string)
            called_url = mock_get.call_args[1]['url']
            assert "a=1" in called_url

            # Test HTTPError raising
            mock_response.status_code = 404
            mock_response.reason = "Not Found"
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {})

    # Test Case 2: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"urllib content"
            # urlopen returns a file-like object, we simulate it
            mock_urlopen.return_value = mock_response
            
            # Mocking the return of urlopen to behave like a stream
            # Since _urllib returns the result of urlopen directly
            result = url_opener("http://example.com", {'method': 'GET'})
            
            assert mock_urlopen.called
            assert result == mock_response

    # Test Case 3: Testing _query logic specifically for URL construction
    # (Checking the edge cases of string concatenation in _query)
    kwargs = {'data': {'id': '123'}}
    url_with_q = "http://test.com?existing=true"
    url_no_q = "http://test.com"
    url_with_amp = "http://test.com?"

    # Test appending with &
    u, d = _query(url_with_q, 'get', kwargs.copy())
    assert u == "http://test.com?existing=true&id=123"

    # Test appending with ?
    u, d = _query(url_no_q, 'get', kwargs.copy())
    assert u == "http://test.com?id=123"

    # Test appending when ? is already at the end
    u, d = _query(url_with_amp, 'get', kwargs.copy())
    assert u == "http://test.com?id=123"

    # Test POST-like behavior (data remains as bytes and URL doesn't change)
    kwargs_post = {'data': {'key': 'val'}}
    u, d = _query("http://test.com", 'POST', kwargs_post)
    assert u == "http://test.com"
    assert b"key=val" in d
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Successful requests-based GET request (with data)
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success_html"
        mock_response.url = "http://example.com?a=b"
        mock_get.return_value = mock_response
        
        kwargs = {'data': {'a': 'b'}, 'timeout': 10}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "success_html"
        # Check if data was encoded into query params in the URL
        args, kwargs_called = mock_get.call_args
        assert "a=b" in kwargs_called['url']

    # Test Case 2: Successful requests-based POST request with headers
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.post') as mock_post:
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "created"
        mock_post.return_value = mock_response
        
        kwargs = {'data': 'raw_body', 'headers': {'User-Agent': 'test'}}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "created"
        args, kwargs_called = mock_post.call_args
        assert kwargs_called['headers'] == {'User-Agent': 'test'}
        assert kwargs_called['data'] == b'raw_body'

    # Test Case 3: HTTP Error handling in requests
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {})

    # Test Case 4: urllib fallback (when HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        kwargs = {'method': 'GET', 'data': {'key': 'val'}}
        url_opener("http://example.com", kwargs)
        
        args, kwargs_called = mock_urlopen.call_args
        assert "key=val" in args[0]
        assert args[1] is None  # because _query pops data if it's a dict/list and converts to urlencode

    # Test Case 5: URL query string concatenation (existing params)
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        # Test adding & to existing query params
        url_opener("http://example.com?existing=1", {'data': {'new': '2'}})
        args, kwargs_called = mock_get.call_args
        assert "http://example.com?existing=1&new=2" in kwargs_called['url']

    # Test Case 6: Encoding parameter application
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "encoded_content"
        mock_get.return_value = mock_response
        
        url_opener("http://example.com", {'encoding': 'utf-16'})
        assert mock_response.encoding == 'utf-16'
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url, expected_data", [
    ("GET", {"data": {"key": "val"}}, "http://test.com?key=val", None),
    ("GET", {"data": [("a", "1"), ("b", "2")]}, "http://test.com?a=1&b=2", None),
    ("POST", {"data": {"key": "val"}}, "http://test.com", b"key=val"),
    ("GET", {}, "http://test.com", None),
])
def test_query_logic(method, params, expected_url, expected_data):
    kwargs = params.copy()
    url, data = _query("http://test.com", method, kwargs)
    assert url == expected_url
    if expected_data is None:
        assert data is None
    else:
        assert data == expected_data

def test_url_opener_requests_success():
    with patch("HAS_REQUEST", True), \
         patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        result = url_opener("http://example.com", {"method": "get"})
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()

def test_url_opener_requests_error():
    with patch("HAS_REQUEST", True), \
         patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "http://example.com"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {"method": "get"})

def test_url_opener_urllib_success():
    with patch("HAS_REQUEST", False), \
         patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        # urllib response behaves like a file/buffer
        mock_response.read.return_value = b"html_content"
        mock_urlopen.return_value = mock_response
        
        # Note: urlopen returns the object, we need to check if it was called correctly
        result = url_opener("http://example.com", {"method": "get"})
        
        assert mock_urlopen.called
        args, kwargs = mock_urlopen.call_args
        assert args[0] == "http://example.com"

def test_url_opener_with_session():
    with patch("HAS_REQUEST", True), \
         patch("requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_instance = mock_session_class.return_value
        mock_session_instance.get.return_value.status_code = 200
        mock_session_instance.get.return_value.text = "session_content"
        
        kwargs = {"session": mock_session_instance, "method": "get"}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_content"
        mock_session_instance.get.assert_called()

def test_url_opener_with_headers_and_auth():
    with patch("HAS_REQUEST", True), \
         patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        kwargs = {
            "method": "get",
            "headers": {"Authorization": "Bearer token"},
            "auth": ("user", "pass")
        }
        url_opener("http://example.com", kwargs)
        
        # Check if allowed_args were passed to requests
        args, kwargs_passed = mock_get.call_args
        assert kwargs_passed['headers'] == {"Authorization": "Bearer token"}
        assert kwargs_passed['auth'] == ("user", "pass")

def test_url_opener_encoding():
    with patch("HAS_REQUEST", True), \
         patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "utf8_content"
        mock_get.return_value = mock_response
        
        url_opener("http://example.com", {"encoding": "utf-8"})
        
        assert mock_response.encoding == "utf-8"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test cases for _query logic via url_opener
    test_url = "http://example.com"
    
    # 1. Test GET with dict data (should append to URL)
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_get.return_value = mock_response
        
        kwargs = {'data': {'key': 'val'}, 'method': 'get'}
        result = url_opener(test_url, kwargs)
        
        assert result == "success"
        # Verify URL was transformed: http://example.com?key=val
        args, kwargs_called = mock_get.call_args
        assert "key=val" in kwargs_called['url']

    # 2. Test GET with existing query params (should use &)
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_get.return_value = mock_response
        
        kwargs = {'data': {'a': 'b'}, 'method': 'get'}
        url_with_query = "http://example.com?existing=true"
        url_opener(url_with_query, kwargs)
        
        args, kwargs_called = mock_get.call_args
        assert "existing=true&a=b" in kwargs_called['url']

    # 3. Test HTTPError raising for non-2xx status codes
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = test_url
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener(test_url, {'method': 'get'})

    # 4. Test allowed_args passing (e.g., headers)
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_get.return_value = mock_response
        
        headers = {'User-Agent': 'test'}
        url_opener(test_url, {'method': 'get', 'headers': headers})
        
        _, kwargs_called = mock_get.call_args
        assert kwargs_called['headers'] == headers

    # 5. Test urllib fallback (when HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value = mock_response
            
            url_opener(test_url, {'method': 'get'})
            
            # Verify urlopen was called with the correct URL and data (encoded)
            args, kwargs_called = mock_urlopen.call_args
            assert args[0] == test_url
            assert kwargs_called['timeout'] == 60

    # 6. Test encoding parameter in requests
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "utf8-content"
        mock_get.return_value = mock_response
        
        result = url_opener(test_url, {'method': 'get', 'encoding': 'utf-8'})
        assert result == "utf8-content"
        assert mock_response.encoding == 'utf-8'

    # 7. Test session usage in requests
    with patch('requests.get') as mock_get:
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(status_code=200, text="session-ok")
        
        url_opener(test_url, {'method': 'get', 'session': mock_session})
        
        mock_session.get.assert_called()

    # 8. Test POST method (data should be passed in body, not URL)
    with patch('requests.get') as mock_post: # Note: code uses getattr(requests, method)
        # We patch 'requests.post' specifically if it exists or the dynamic call
        # The implementation calls getattr(requests, str(method))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "post-ok"
        with patch('requests.post', return_value=mock_response):
            kwargs = {'data': 'raw_payload', 'method': 'post'}
            url_opener(test_url, kwargs)
            
            args, kwargs_called = mock_post.call_args
            assert kwargs_called['url'] == test_url
            # The code encodes data to bytes for non-GET methods
            assert kwargs_called['data'] == b'raw_payload'
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {"method": "get", "encoding": "utf-8"})
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()

        # Test Case 2: requests raises HTTPError for non-2xx status
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {"method": "get"})

    # Test Case 3: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urllib_response = MagicMock()
        mock_urllib_response.read.return_value = b"<html>Urllib</html>"
        
        with patch('__main__.urlopen', return_value=mock_urllib_response) as mock_urlopen:
            result = url_opener("http://example.com", {"method": "get", "data": {"key": "val"}})
            # urllib returns the response object, we check if urlopen was called correctly
            assert mock_urlopen.called
            args, kwargs = mock_urlopen.call_args
            assert "http://example.com?key=val" in args[0]

    # Test Case 4: _query logic for GET with data (params in URL)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        with patch('requests.get', return_value=mock_response) as mock_get:
            # Test appending data to URL via _query logic in _requests
            url_opener("http://example.com", {"method": "get", "data": {"a": 1}})
            called_url = mock_get.call_args[1]['url']
            assert "a=1" in called_url

    # Test Case 5: Session usage in requests
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.text = "session_ok"
        
        with patch('requests.get') as mock_req_get:
            result = url_opener("http://example.com", {"method": "get", "session": mock_session})
            assert result == "session_ok"
            mock_session.get.assert_called_once()

    # Test Case 6: Verifying allowed_args pass through to requests
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        kwargs = {'method': 'get', 'headers': {'User-Agent': 'test'}, 'timeout': 10}
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", kwargs)
            # Check if headers and timeout were passed to the request
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['headers'] == {'User-Agent': 'test'}
            assert call_kwargs['timeout'] == 10
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.parametrize("method, params, expected_url", [
    ("GET", {"data": {"key": "val"}}, "http://test.com?key=val"),
    ("GET", {"data": [("a", "1"), ("b", "2")]}, "http://test.com?a=1&b=2"),
    ("POST", {"data": {"key": "val"}}, "http://test.com"),
    ("GET", {}, "http://test.com"),
])
def test_query_logic(method, params, expected_url):
    kwargs = {'method': method}
    if 'data' in params:
        kwargs['data'] = params['data']
    
    url, data = _query("http://test.com", method, kwargs)
    assert url == expected_url
    if 'data' in params and isinstance(params['data'], (dict, list, tuple)):
        assert data is None

def test_url_opener_requests_success():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>success</html>"
        mock_get.return_value = mock_response
        
        kwargs = {'method': 'get', 'timeout': 10, 'headers': {'User-Agent': 'test'}}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "<html>success</html>"
        mock_get.assert_called_once()
        args, kwargs_call = mock_get.call_args
        assert kwargs_call['timeout'] == 10
        assert kwargs_call['headers'] == {'User-Agent': 'test'}

def test_url_opener_requests_http_error():
    with patch('__main__.HAS_REQUEST', True), \
         patch('requests.get') as mock_get:
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with pytest.raises(HTTPError):
            url_opener("http://example.com", {'method': 'get'})

def test_url_opener_urllib_success():
    with patch('__main__.HAS_REQUEST', False), \
         patch('urllib.request.urlopen') as mock_urlopen:
        
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        
        url_opener("http://example.com", {'method': 'get'})
        
        mock_urlopen.assert_called_once()
        args, kwargs = mock_urlopen.call_args
        assert args[0] == "http://example.com"
        assert kwargs['timeout'] == DEFAULT_TIMEOUT

def test_url_opener_requests_with_session():
    with patch('__main__.HAS_REQUEST', True):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "session_content"
        mock_session.get.return_value = mock_response
        
        kwargs = {'method': 'get', 'session': mock_session}
        result = url_opener("http://example.com", kwargs)
        
        assert result == "session_content"
        mock_session.get.assert_called_once()

def test_query_url_append_existing_params():
    kwargs = {'method': 'GET', 'data': {'b': '2'}}
    url, _ = _query("http://test.com?a=1", "GET", kwargs)
    assert url == "http://test.com?a=1&b=2"

def test_query_url_append_fragmentary_params():
    kwargs = {'method': 'GET', 'data': {'b': '2'}}
    url, _ = _query("http://test.com?", "GET", kwargs)
    assert url == "http://test.com?b=2"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Successful _requests implementation (when HAS_REQUEST is True)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"
        mock_response.url = "http://example.com"
        mock_response.reason = "OK"
        mock_response.headers = {}

        with patch('requests.get', return_value=mock_response) as mock_get:
            kwargs = {'method': 'get', 'timeout': 10, 'data': {'key': 'val'}}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            # Check if _query correctly appended data to URL
            mock_get.assert_called_once()
            args, kwargs_call = mock_get.call_args
            assert "key=val" in kwargs_call['url']

    # Test Case 2: _requests raises HTTPError on non-2xx status
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.url = "http://example.com"
        mock_response.reason = "Not Found"
        mock_response.headers = {}
        
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(HTTPError):
                url_opener("http://example.com", {'method': 'get'})

    # Test Case 3: Successful _urllib implementation (when HAS_REQUEST is False)
    with patch('__main__.HAS_REQUEST', False):
        mock_response = MagicMock()
        with patch('urllib.request.urlopen', return_value=mock_response) as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = b"content" 
            # Note: urlopen returns a file-like object; we simulate the return
            
            kwargs = {'method': 'get', 'data': {'a': 'b'}}
            url_opener("http://example.com", kwargs)
            
            args, kwargs_call = mock_urlopen.call_args
            assert args[0] == "http://example.com?a=b"
            assert kwargs_call['timeout'] == 60

    # Test Case 4: Verifying allowed_args are passed to requests
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        
        kwargs = {
            'method': 'post',
            'headers': {'User-Agent': 'test'},
            'auth': ('user', 'pass'),
            'data': 'raw_data'
        }
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            url_opener("http://example.com", kwargs)
            args, kwargs_call = mock_post.call_args
            assert kwargs_call['headers'] == {'User-Agent': 'test'}
            assert kwargs_call['auth'] == ('user', 'pass')
            assert kwargs_call['data'] == b'raw_data'

    # Test Case 5: _query URL manipulation (adding ? or &)
    with patch('__main__.HAS_REQUEST', True):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        
        # Case: Existing query param, needs &
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com?existing=true", {'method': 'get', 'data': {'new': 'val'}})
            assert "existing=true&new=val" in mock_get.call_args[1]['url']

        # Case: No query param, needs ?
        with patch('requests.get', return_value=mock_response) as mock_get:
            url_opener("http://example.com", {'method': 'get', 'data': {'new': 'val'}})
            assert "http://example.com?new=val" in mock_get.call_args[1]['url']
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock, patch

def test_url_opener():
    # Test Case 1: Using requests (HAS_REQUEST = True)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html>Success</html>"
        mock_resp.url = "http://example.com"
        mock_response_method = MagicMock(return_value=mock_resp)
        
        with patch('requests.get', mock_response_method) as mock_get:
            kwargs = {'timeout': 10, 'headers': {'User-Agent': 'test'}}
            result = url_opener("http://example.com", kwargs)
            
            assert result == "<html>Success</html>"
            mock_get.assert_called_once()
            # Verify allowed_args filtering (headers should be passed, but encoding is not in allowed_args)
            args, kwargs_call = mock_get.call_args
            assert 'headers' in kwargs_call
            assert 'encoding' not in kwargs_call

    # Test Case 2: Using requests with error (HTTPError raised)
    with patch('__main__.HAS_REQUEST', True):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"
        mock_resp.url = "http://example.com/404"
        mock_resp.headers = {}
        
        with patch('requests.get', return_value=mock_resp):
            with pytest.raises(HTTPError):
                url_opener("http://example.com/404", {})

    # Test Case 3: Using urllib (HAS_REQUEST = False)
    with patch('__main__.HAS_REQUEST', False):
        mock_urlopen = MagicMock()
        # urlopen returns a file-like object with .read()
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>urllib</html>"
        mock_urlopen.return_value = mock_response
        
        with patch('__main__.urlopen', mock_urlopen) as mock_url_open:
            kwargs = {'method': 'get', 'data': {'key': 'val'}}
            # Note: _urllib returns the urlopen object itself, so we check its behavior
            result = url_opener("http://example.com", kwargs)
            
            # Check if query string was appended correctly for GET
            expected_url = "http://example.com?key=val"
            args, kwargs_call = mock_url_open.call_args
            assert args[0] == expected_url
            assert result.read() == b"<html>urllib</html>"

    # Test Case 4: _query logic for GET with existing params
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.get') as mock_get:
            mock_resp = MagicMock(status_code=200, text="ok")
            mock_get.return_value = mock_resp
            
            # Test appending '&' if '?' already exists
            url_opener("http://example.com?a=1", {'data': {'b': 2}, 'method': 'get'})
            args, kwargs_call = mock_get.call_args
            assert "http://example.com?a=1&b=2" in args[0]

    # Test Case 5: _query logic for POST (data remains as encoded bytes)
    with patch('__main__.HAS_REQUEST', True):
        with patch('requests.post') as mock_post:
            mock_resp = MagicMock(status_code=200, text="ok")
            mock_post.return_value = mock_resp
            
            kwargs = {'data': {'key': 'value'}, 'method': 'post'}
            url_opener("http://example.com", kwargs)
            args, kwargs_call = mock_post.call_args
            # data should be encoded to bytes
            assert kwargs_call['data'] == b'key=value'
```


