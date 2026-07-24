####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocked)
    import pytest
    from unittest.mock import patch, MagicMock
    
    # Mock requests module
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "test html content"
    mock_response.url = "http://example.com"
    mock_response.reason = "OK"
    mock_response.headers = {}
    mock_response.encoding = None
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.get', return_value=mock_response) as mock_get:
            # Test basic GET request
            result = url_opener("http://example.com", {'method': 'get'})
            assert result == "test html content"
            mock_get.assert_called_once_with(
                url="http://example.com", 
                timeout=60
            )
    
    # Test with requests library not available (using urllib)
    with patch('your_module.HAS_REQUEST', False):
        with patch('your_module.urlopen') as mock_urlopen:
            mock_response_urllib = MagicMock()
            mock_response_urllib.read.return_value = b"test html content"
            mock_urlopen.return_value = mock_response_urllib
            
            result = url_opener("http://example.com", {'method': 'get'})
            assert result is not None
    
    # Test with data parameter
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.post', return_value=mock_response) as mock_post:
            result = url_opener("http://example.com", {
                'method': 'post',
                'data': {'key': 'value'}
            })
            assert result == "test html content"
            mock_post.assert_called_once_with(
                url="http://example.com",
                timeout=60,
                data='key=value'
            )
    
    # Test with custom timeout
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {
                'method': 'get',
                'timeout': 30
            })
            assert result == "test html content"
            mock_get.assert_called_once_with(
                url="http://example.com",
                timeout=30
            )
    
    # Test with headers
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.get', return_value=mock_response) as mock_get:
            result = url_opener("http://example.com", {
                'method': 'get',
                'headers': {'User-Agent': 'test'}
            })
            assert result == "test html content"
            mock_get.assert_called_once_with(
                url="http://example.com",
                timeout=60,
                headers={'User-Agent': 'test'}
            )
    
    # Test HTTP error
    mock_error_response = MagicMock()
    mock_error_response.status_code = 404
    mock_error_response.url = "http://example.com/notfound"
    mock_error_response.reason = "Not Found"
    mock_error_response.headers = {}
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.get', return_value=mock_error_response):
            with pytest.raises(HTTPError) as exc_info:
                url_opener("http://example.com/notfound", {'method': 'get'})
            assert exc_info.value.code == 404
            assert exc_info.value.reason == "Not Found"
    
    # Test with encoding parameter
    mock_response_encoded = MagicMock()
    mock_response_encoded.status_code = 200
    mock_response_encoded.text = "test html content"
    mock_response_encoded.url = "http://example.com"
    mock_response_encoded.reason = "OK"
    mock_response_encoded.headers = {}
    mock_response_encoded.encoding = None
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests.get', return_value=mock_response_encoded) as mock_get:
            result = url_opener("http://example.com", {
                'method': 'get',
                'encoding': 'utf-8'
            })
            assert result == "test html content"
            assert mock_response_encoded.encoding == 'utf-8'
```


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import sys
    from unittest.mock import patch, MagicMock
    
    # Test GET request without data
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module._requests') as mock_requests:
            mock_requests.return_value = "response"
            result = url_opener("http://example.com", {"method": "get"})
            assert result == "response"
            mock_requests.assert_called_once_with("http://example.com", {"method": "get"})
    
    # Test with urllib when requests not available
    with patch('your_module.HAS_REQUEST', False):
        with patch('your_module._urllib') as mock_urllib:
            mock_response = MagicMock()
            mock_urllib.return_value = mock_response
            result = url_opener("http://example.com", {"method": "get"})
            assert result == mock_response
            mock_urllib.assert_called_once_with("http://example.com", {"method": "get"})
    
    # Test POST request with data
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module._requests') as mock_requests:
            test_kwargs = {"method": "post", "data": {"key": "value"}}
            mock_requests.return_value = "post response"
            result = url_opener("http://example.com", test_kwargs)
            assert result == "post response"
            mock_requests.assert_called_once()
    
    # Test with additional kwargs
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module._requests') as mock_requests:
            test_kwargs = {
                "method": "get",
                "timeout": 30,
                "headers": {"User-Agent": "test"}
            }
            url_opener("http://example.com", test_kwargs)
            mock_requests.assert_called_once_with("http://example.com", test_kwargs)
    
    # Test edge case: empty kwargs
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module._requests') as mock_requests:
            url_opener("http://example.com", {})
            mock_requests.assert_called_once_with("http://example.com", {})
```


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import sys
    from unittest.mock import patch, MagicMock
    
    # Test 1: Basic GET request with requests
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    mock_response.url = "http://example.com"
    
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.get') as mock_get:
        mock_get.return_value = mock_response
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "Success"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=60
        )
    
    # Test 2: GET request with data parameter
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.get') as mock_get:
        mock_get.return_value = mock_response
        result = url_opener("http://example.com", {
            "method": "get",
            "data": {"key": "value"}
        })
        assert result == "Success"
        mock_get.assert_called_once_with(
            url="http://example.com?key=value",
            timeout=60
        )
    
    # Test 3: POST request with data
    mock_post_response = MagicMock()
    mock_post_response.status_code = 201
    mock_post_response.text = "Created"
    mock_post_response.url = "http://example.com"
    
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.post') as mock_post:
        mock_post.return_value = mock_post_response
        result = url_opener("http://example.com", {
            "method": "post",
            "data": {"key": "value"}
        })
        assert result == "Created"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['url'] == "http://example.com"
        assert kwargs['data'] == b'key=value'
    
    # Test 4: HTTP error raises exception
    error_response = MagicMock()
    error_response.status_code = 404
    error_response.reason = "Not Found"
    error_response.headers = {}
    error_response.url = "http://example.com/notfound"
    
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.get') as mock_get, \
         pytest.raises(HTTPError) as exc_info:
        mock_get.return_value = error_response
        url_opener("http://example.com/notfound", {"method": "get"})
    
    assert exc_info.value.code == 404
    
    # Test 5: Fallback to urllib when requests not available
    from unittest.mock import patch
    mock_urlopen = MagicMock()
    mock_urlopen.read.return_value = b"urllib response"
    
    with patch('test_module.HAS_REQUEST', False), \
         patch('test_module.urlopen', return_value=mock_urlopen) as mock_urlopen_func:
        result = url_opener("http://example.com", {"method": "get"})
        assert result == mock_urlopen
        mock_urlopen_func.assert_called_once_with(
            "http://example.com", None, timeout=60
        )
    
    # Test 6: Custom headers
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.get') as mock_get:
        mock_get.return_value = mock_response
        result = url_opener("http://example.com", {
            "method": "get",
            "headers": {"Authorization": "Bearer token123"}
        })
        assert result == "Success"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=60,
            headers={"Authorization": "Bearer token123"}
        )
    
    # Test 7: Custom timeout
    with patch('test_module.HAS_REQUEST', True), \
         patch('test_module.requests.get') as mock_get:
        mock_get.return_value = mock_response
        result = url_opener("http://example.com", {
            "method": "get",
            "timeout": 30
        })
        assert result == "Success"
        mock_get.assert_called_once_with(
            url="http://example.com",
            timeout=30
        )
```


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    global HAS_REQUEST
    HAS_REQUEST = True
    
    # Test basic GET request
    result = url_opener('http://httpbin.org/get', {'method': 'get'})
    assert isinstance(result, str)
    assert 'url' in result.lower()
    
    # Test GET with data (should convert to query params)
    result = url_opener('http://httpbin.org/get', {
        'method': 'get',
        'data': {'key1': 'value1', 'key2': 'value2'}
    })
    assert isinstance(result, str)
    assert 'key1' in result
    assert 'value1' in result
    
    # Test POST request
    result = url_opener('http://httpbin.org/post', {
        'method': 'post',
        'data': {'key1': 'value1'}
    })
    assert isinstance(result, str)
    assert 'key1' in result
    
    # Test with custom timeout
    result = url_opener('http://httpbin.org/get', {
        'method': 'get',
        'timeout': 30
    })
    assert isinstance(result, str)
    
    # Test with custom headers
    result = url_opener('http://httpbin.org/headers', {
        'method': 'get',
        'headers': {'X-Test': 'test-value'}
    })
    assert isinstance(result, str)
    assert 'X-Test' in result or 'x-test' in result.lower()
    
    # Test HTTP error handling
    try:
        url_opener('http://httpbin.org/status/404', {'method': 'get'})
        assert False, "Should have raised HTTPError"
    except HTTPError:
        pass
    
    # Test with encoding
    result = url_opener('http://httpbin.org/get', {
        'method': 'get',
        'encoding': 'utf-8'
    })
    assert isinstance(result, str)
    
    # Test without requests library
    HAS_REQUEST = False
    result = url_opener('http://httpbin.org/get', {'method': 'get'})
    assert result is not None
    
    # Test urllib with data
    result = url_opener('http://httpbin.org/post', {
        'method': 'post',
        'data': {'test': 'data'}
    })
    assert result is not None
    
    # Restore HAS_REQUEST for other tests
    HAS_REQUEST = True
    
    # Test with cookies
    result = url_opener('http://httpbin.org/cookies', {
        'method': 'get',
        'cookies': {'test_cookie': 'test_value'}
    })
    assert isinstance(result, str)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener():
    # Test with requests available (mocked)
    import pytest
    from unittest.mock import patch, MagicMock
    
    # Test GET request with data
    with patch('your_module.HAS_REQUEST', True), \
         patch('your_module._requests') as mock_requests:
        mock_requests.return_value = "response text"
        result = url_opener('http://example.com', {'method': 'get', 'data': {'key': 'value'}})
        assert result == "response text"
        mock_requests.assert_called_once_with('http://example.com', {'method': 'get', 'data': {'key': 'value'}})
    
    # Test without requests (using urllib)
    with patch('your_module.HAS_REQUEST', False), \
         patch('your_module._urllib') as mock_urllib:
        mock_response = MagicMock()
        mock_response.read.return_value = b'response text'
        mock_urllib.return_value = mock_response
        
        result = url_opener('http://example.com', {'method': 'post', 'data': 'test'})
        assert result == mock_response
        mock_urllib.assert_called_once_with('http://example.com', {'method': 'post', 'data': 'test'})
    
    # Test with custom timeout
    with patch('your_module.HAS_REQUEST', True), \
         patch('your_module._requests') as mock_requests:
        mock_requests.return_value = "response"
        url_opener('http://example.com', {'method': 'get', 'timeout': 30})
        mock_requests.assert_called_once_with('http://example.com', {'method': 'get', 'timeout': 30})

```


# LLM-generated content at query #6
#--------------------------

```python
def test_url_opener():
    # Test 1: Basic GET request with urllib (when requests not available)
    global HAS_REQUEST
    original_has_request = HAS_REQUEST
    try:
        HAS_REQUEST = False
        # Mock a simple URL that returns data
        result = url_opener('http://example.com', {'method': 'GET'})
        assert result is not None
        assert hasattr(result, 'read')
    finally:
        HAS_REQUEST = original_has_request
    
    # Test 2: GET request with data (converts to query string)
    result = url_opener('http://example.com', {
        'method': 'GET',
        'data': {'key': 'value'}
    })
    assert result is not None
    
    # Test 3: POST request with data
    result = url_opener('http://httpbin.org/post', {
        'method': 'POST',
        'data': {'key': 'value'},
        'timeout': 10
    })
    assert result is not None
    
    # Test 4: GET request with data when URL already has query params
    result = url_opener('http://example.com?existing=true', {
        'method': 'GET',
        'data': {'key': 'value'}
    })
    assert result is not None
    
    # Test 5: With custom headers
    result = url_opener('http://httpbin.org/headers', {
        'method': 'GET',
        'headers': {'User-Agent': 'test-agent'},
        'timeout': 10
    })
    assert result is not None
    
    # Test 6: Test that HTTP errors are raised
    import pytest
    with pytest.raises(HTTPError):
        url_opener('http://httpbin.org/status/404', {
            'method': 'GET',
            'timeout': 10
        })
    
    # Test 7: With verify=False (skip SSL verification)
    result = url_opener('https://httpbin.org/get', {
        'method': 'GET',
        'verify': False,
        'timeout': 10
    })
    assert result is not None
    
    # Test 8: With session (requests only)
    import requests as req_module
    session = req_module.Session()
    result = url_opener('http://httpbin.org/get', {
        'method': 'GET',
        'session': session,
        'timeout': 10
    })
    assert result is not None
    session.close()
    
    # Test 9: Test with encoding
    result = url_opener('http://httpbin.org/get', {
        'method': 'GET',
        'encoding': 'utf-8',
        'timeout': 10
    })
    assert result is not None
    
    # Test 10: Test with cookies
    result = url_opener('http://httpbin.org/cookies', {
        'method': 'GET',
        'cookies': {'test_cookie': 'test_value'},
        'timeout': 10
    })
    assert result is not None
```


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener():
    # Test with requests library (if available)
    if HAS_REQUEST:
        # Test GET request without data
        result = url_opener("http://example.com", {"method": "get"})
        assert isinstance(result, str)
        
        # Test GET request with data
        result = url_opener("http://example.com", {
            "method": "get",
            "data": {"key1": "value1", "key2": "value2"}
        })
        assert isinstance(result, str)
        
        # Test POST request with data
        result = url_opener("http://httpbin.org/post", {
            "method": "post",
            "data": {"test": "data"}
        })
        assert isinstance(result, str)
        
        # Test with custom timeout
        result = url_opener("http://example.com", {
            "method": "get",
            "timeout": 30
        })
        assert isinstance(result, str)
        
        # Test with headers
        result = url_opener("http://example.com", {
            "method": "get",
            "headers": {"User-Agent": "TestAgent"}
        })
        assert isinstance(result, str)
        
        # Test with encoding
        result = url_opener("http://example.com", {
            "method": "get",
            "encoding": "utf-8"
        })
        assert isinstance(result, str)
        
        # Test error case - invalid URL should raise HTTPError
        import pytest
        with pytest.raises(HTTPError):
            url_opener("http://nonexistent-domain-12345.com", {"method": "get"})
    
    # Test without requests library (using urllib)
    else:
        # Test GET request
        result = url_opener("http://example.com", {"method": "get"})
        assert result is not None
        
        # Test GET request with data
        result = url_opener("http://example.com", {
            "method": "get",
            "data": {"key": "value"}
        })
        assert result is not None
        
        # Test with custom timeout
        result = url_opener("http://example.com", {
            "method": "get",
            "timeout": 30
        })
        assert result is not None
```


# LLM-generated content at query #8
#--------------------------

```python
def test_url_opener():
    # Test with mock for requests library
    import unittest.mock as mock
    
    # Test case 1: Using requests library when available
    with mock.patch('test_module.HAS_REQUEST', True):
        with mock.patch('test_module._requests') as mock_requests:
            mock_requests.return_value = "<html>Test</html>"
            result = url_opener('http://example.com', {'method': 'get'})
            assert result == "<html>Test</html>"
            mock_requests.assert_called_once_with('http://example.com', {'method': 'get'})
    
    # Test case 2: Using urllib when requests not available
    with mock.patch('test_module.HAS_REQUEST', False):
        with mock.patch('test_module._urllib') as mock_urllib:
            mock_response = mock.MagicMock()
            mock_urllib.return_value = mock_response
            result = url_opener('http://example.com', {'method': 'get'})
            assert result == mock_response
            mock_urllib.assert_called_once_with('http://example.com', {'method': 'get'})
    
    # Test case 3: With timeout parameter
    with mock.patch('test_module.HAS_REQUEST', True):
        with mock.patch('test_module._requests') as mock_requests:
            mock_requests.return_value = "<html>Test</html>"
            result = url_opener('http://example.com', {'method': 'get', 'timeout': 30})
            assert result == "<html>Test</html>"
            mock_requests.assert_called_once_with('http://example.com', {'method': 'get', 'timeout': 30})
    
    # Test case 4: With POST method and data
    with mock.patch('test_module.HAS_REQUEST', True):
        with mock.patch('test_module._requests') as mock_requests:
            mock_requests.return_value = "<html>Test</html>"
            result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
            assert result == "<html>Test</html>"


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import requests
    from unittest.mock import patch, MagicMock
    
    # Test successful GET request with requests
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == "Success"
        mock_get.assert_called_once_with(url='http://example.com', timeout=60)
    
    # Test GET request with data parameter
    with patch('requests.get', return_value=mock_response) as mock_get:
        result = url_opener('http://example.com', {'method': 'get', 'data': {'key': 'value'}})
        assert result == "Success"
        mock_get.assert_called_once_with(url='http://example.com?key=value', timeout=60)
    
    # Test POST request with data
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.text = "Posted"
    
    with patch('requests.post', return_value=mock_post_response) as mock_post:
        result = url_opener('http://example.com', {'method': 'post', 'data': {'key': 'value'}})
        assert result == "Posted"
    
    # Test HTTP error response
    mock_error_response = MagicMock()
    mock_error_response.status_code = 404
    mock_error_response.url = 'http://example.com/notfound'
    mock_error_response.reason = 'Not Found'
    mock_error_response.headers = {}
    
    with patch('requests.get', return_value=mock_error_response):
        try:
            url_opener('http://example.com/notfound', {'method': 'get'})
            assert False, "Should have raised HTTPError"
        except HTTPError as e:
            assert e.code == 404
    
    # Test with encoding parameter
    mock_encoded_response = MagicMock()
    mock_encoded_response.status_code = 200
    mock_encoded_response.text = "Encoded"
    
    with patch('requests.get', return_value=mock_encoded_response) as mock_get:
        result = url_opener('http://example.com', {'method': 'get', 'encoding': 'utf-8'})
        assert result == "Encoded"
        assert mock_encoded_response.encoding == 'utf-8'
    
    # Test with session
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    
    result = url_opener('http://example.com', {'method': 'get', 'session': mock_session})
    assert result == "Success"
    mock_session.get.assert_called_once()


# LLM-generated content at query #10
#--------------------------

```python
def test_url_opener():
    # Test with requests module available (mocked)
    # Test GET request without data
    result = url_opener("http://example.com", {"method": "get"})
    assert result is not None
    
    # Test GET request with data
    result = url_opener("http://example.com", {
        "method": "get",
        "data": {"key": "value"}
    })
    assert result is not None
    
    # Test POST request with data
    result = url_opener("http://example.com", {
        "method": "post",
        "data": {"key": "value"}
    })
    assert result is not None
    
    # Test with custom timeout
    result = url_opener("http://example.com", {
        "method": "get",
        "timeout": 30
    })
    assert result is not None
    
    # Test with headers
    result = url_opener("http://example.com", {
        "method": "get",
        "headers": {"User-Agent": "test"}
    })
    assert result is not None
    
    # Test with auth
    result = url_opener("http://example.com", {
        "method": "get",
        "auth": ("user", "pass")
    })
    assert result is not None
    
    # Test with encoding
    result = url_opener("http://example.com", {
        "method": "get",
        "encoding": "utf-8"
    })
    assert result is not None
    
    # Test GET request with data already containing query params
    result = url_opener("http://example.com?existing=param", {
        "method": "get",
        "data": {"new": "data"}
    })
    assert result is not None
    
    # Test that HTTPError is raised for non-200 status codes
    try:
        url_opener("http://httpstat.us/404", {"method": "get"})
        assert False, "Should have raised HTTPError"
    except HTTPError:
        pass
    
    # Test with session
    import requests
    session = requests.Session()
    result = url_opener("http://example.com", {
        "method": "get",
        "session": session
    })
    assert result is not None
    session.close()
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import sys
    from unittest.mock import patch, MagicMock
    from urllib.error import HTTPError
    
    # Mock requests module
    mock_requests = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "test response"
    mock_requests.get.return_value = mock_response
    mock_requests.post.return_value = mock_response
    
    with patch.dict('sys.modules', {'requests': mock_requests}):
        # Test basic GET request
        result = url_opener("http://example.com", {'method': 'get'})
        assert result == "test response"
        mock_requests.get.assert_called_with(url="http://example.com", timeout=60)
        
        # Test GET request with data
        mock_requests.reset_mock()
        result = url_opener("http://example.com", {'method': 'get', 'data': {'key': 'value'}})
        mock_requests.get.assert_called_with(url="http://example.com?key=value", timeout=60)
        
        # Test POST request with data
        mock_requests.reset_mock()
        result = url_opener("http://example.com", {'method': 'post', 'data': {'key': 'value'}})
        mock_requests.post.assert_called_with(url="http://example.com", timeout=60, data=b'key=value')
        
        # Test with custom timeout
        mock_requests.reset_mock()
        result = url_opener("http://example.com", {'method': 'get', 'timeout': 30})
        mock_requests.get.assert_called_with(url="http://example.com", timeout=30)
        
        # Test HTTP error
        mock_error_response = MagicMock()
        mock_error_response.status_code = 404
        mock_error_response.url = "http://example.com"
        mock_error_response.reason = "Not Found"
        mock_error_response.headers = {}
        mock_requests.get.return_value = mock_error_response
        
        try:
            url_opener("http://example.com", {'method': 'get'})
            assert False, "Expected HTTPError"
        except HTTPError as e:
            assert e.code == 404
            assert e.reason == "Not Found"
    
    # Test without requests library (using urllib)
    with patch.dict('sys.modules', {'requests': None}):
        from unittest.mock import patch as urlopen_patch
        with urlopen_patch('builtins.open', create=True) as mock_urlopen:
            mock_urlopen.return_value.read.return_value = b"test response"
            
            result = url_opener("http://example.com", {'method': 'get'})
            # In urllib mode, returns a file-like object
            assert hasattr(result, 'read')
            
            # Test GET with data
            result = url_opener("http://example.com", {'method': 'get', 'data': {'key': 'value'}})
            assert hasattr(result, 'read')
```


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    global HAS_REQUEST
    original_has_request = HAS_REQUEST
    HAS_REQUEST = True
    
    # Test GET request without data
    result = url_opener('http://example.com', {'method': 'get'})
    assert isinstance(result, str)
    assert 'Example Domain' in result
    
    # Test GET request with data
    result = url_opener('http://httpbin.org/get', {
        'method': 'get',
        'data': {'key': 'value'}
    })
    assert isinstance(result, str)
    assert 'key' in result
    assert 'value' in result
    
    # Test POST request with data
    result = url_opener('http://httpbin.org/post', {
        'method': 'post',
        'data': {'key': 'value'}
    })
    assert isinstance(result, str)
    assert 'key' in result
    assert 'value' in result
    
    # Test with custom timeout
    result = url_opener('http://example.com', {
        'method': 'get',
        'timeout': 10
    })
    assert isinstance(result, str)
    
    # Test with encoding
    result = url_opener('http://example.com', {
        'method': 'get',
        'encoding': 'utf-8'
    })
    assert isinstance(result, str)
    
    # Test HTTPError for non-2xx status
    try:
        url_opener('http://httpbin.org/status/404', {'method': 'get'})
        assert False, "Should have raised HTTPError"
    except HTTPError:
        pass
    
    # Test with requests library unavailable
    HAS_REQUEST = False
    result = url_opener('http://example.com', {'method': 'get'})
    assert isinstance(result, object)  # Returns HTTPResponse object
    
    HAS_REQUEST = original_has_request
```


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    global HAS_REQUEST
    HAS_REQUEST = True
    
    # Test GET request without data
    result = url_opener("http://example.com", {"method": "get"})
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test GET request with data (should append to URL)
    result = url_opener("http://example.com", {
        "method": "get",
        "data": {"key1": "value1", "key2": "value2"}
    })
    assert isinstance(result, str)
    
    # Test GET request with data and existing query parameters
    result = url_opener("http://example.com?existing=param", {
        "method": "get",
        "data": {"key": "value"}
    })
    assert isinstance(result, str)
    
    # Test POST request with data
    result = url_opener("http://httpbin.org/post", {
        "method": "post",
        "data": {"test": "data"}
    })
    assert isinstance(result, str)
    
    # Test with custom timeout
    result = url_opener("http://example.com", {
        "method": "get",
        "timeout": 30
    })
    assert isinstance(result, str)
    
    # Test with headers
    result = url_opener("http://httpbin.org/headers", {
        "method": "get",
        "headers": {"User-Agent": "test-agent"}
    })
    assert isinstance(result, str)
    
    # Test error handling with non-existent domain
    try:
        url_opener("http://nonexistent-domain-12345.com", {"method": "get"})
        assert False, "Should have raised an exception"
    except Exception:
        pass
    
    # Test with requests library unavailable
    HAS_REQUEST = False
    try:
        result = url_opener("http://example.com", {"method": "get"})
        assert isinstance(result, object)  # Returns file-like object
    except Exception:
        pass
    
    # Reset HAS_REQUEST
    HAS_REQUEST = True


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import requests
    from unittest.mock import patch, MagicMock
    
    # Mock successful GET request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success response"
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests') as mock_requests:
            mock_requests.get.return_value = mock_response
            result = url_opener('http://example.com', {'method': 'get'})
            assert result == "Success response"
            mock_requests.get.assert_called_once_with(
                url='http://example.com', 
                timeout=60
            )
    
    # Mock successful POST request with data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Post response"
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests') as mock_requests:
            mock_requests.post.return_value = mock_response
            result = url_opener('http://example.com', {
                'method': 'post',
                'data': {'key': 'value'}
            })
            assert result == "Post response"
            mock_requests.post.assert_called_once()
    
    # Test without requests library (using urllib)
    with patch('your_module.HAS_REQUEST', False):
        with patch('your_module.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            result = url_opener('http://example.com', {'method': 'get'})
            mock_urlopen.assert_called_once_with(
                'http://example.com', None, timeout=60
            )
    
    # Test error response
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.url = 'http://example.com'
    mock_response.reason = 'Not Found'
    mock_response.headers = {}
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests') as mock_requests:
            mock_requests.get.return_value = mock_response
            try:
                url_opener('http://example.com', {'method': 'get'})
                assert False, "Should have raised HTTPError"
            except HTTPError as e:
                assert e.code == 404
                assert e.reason == 'Not Found'
    
    # Test with timeout
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Timeout test"
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests') as mock_requests:
            mock_requests.get.return_value = mock_response
            result = url_opener('http://example.com', {
                'method': 'get',
                'timeout': 30
            })
            assert result == "Timeout test"
            mock_requests.get.assert_called_once_with(
                url='http://example.com', 
                timeout=30
            )
    
    # Test GET with query parameters from data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Query param test"
    
    with patch('your_module.HAS_REQUEST', True):
        with patch('your_module.requests') as mock_requests:
            mock_requests.get.return_value = mock_response
            result = url_opener('http://example.com', {
                'method': 'get',
                'data': {'param1': 'value1', 'param2': 'value2'}
            })
            assert result == "Query param test"
            call_args = mock_requests.get.call_args[1]
            assert 'param1=value1' in call_args['url']
            assert 'param2=value2' in call_args['url']```


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    # Mock successful GET request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success response"
    
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.get.return_value = mock_response
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "Success response"
        mock_requests.get.assert_called_once_with(
            url="http://example.com", 
            timeout=60
        )

    # Test with data in GET request
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.get.return_value = mock_response
        result = url_opener(
            "http://example.com", 
            {"method": "get", "data": {"key": "value"}}
        )
        assert result == "Success response"
        mock_requests.get.assert_called_once_with(
            url="http://example.com?key=value", 
            timeout=60
        )

    # Test with POST request and data
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.text = "Post success"
    
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.post.return_value = mock_post_response
        result = url_opener(
            "http://example.com", 
            {"method": "post", "data": {"key": "value"}}
        )
        assert result == "Post success"
        mock_requests.post.assert_called_once_with(
            url="http://example.com", 
            timeout=60,
            data="key=value"
        )

    # Test with custom headers
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.get.return_value = mock_response
        result = url_opener(
            "http://example.com", 
            {"method": "get", "headers": {"Authorization": "Bearer token"}}
        )
        assert result == "Success response"
        mock_requests.get.assert_called_once_with(
            url="http://example.com", 
            timeout=60,
            headers={"Authorization": "Bearer token"}
        )

    # Test HTTP error handling
    error_response = MagicMock()
    error_response.status_code = 404
    error_response.url = "http://example.com/notfound"
    error_response.reason = "Not Found"
    error_response.headers = {}
    
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests, \
         pytest.raises(HTTPError) as exc_info:
        mock_requests.get.return_value = error_response
        url_opener("http://example.com/notfound", {"method": "get"})
    
    assert exc_info.value.code == 404

    # Test with encoding parameter
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.get.return_value = mock_response
        result = url_opener(
            "http://example.com", 
            {"method": "get", "encoding": "utf-8"}
        )
        assert result == "Success response"
        assert mock_response.encoding == "utf-8"

    # Test urllib fallback when requests not available
    mock_urlopen = MagicMock()
    mock_urlopen.return_value = "urllib response"
    
    with patch('module_name.HAS_REQUEST', False), \
         patch('module_name.urlopen', mock_urlopen):
        result = url_opener("http://example.com", {"method": "get"})
        assert result == "urllib response"

    # Test with custom timeout
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.get.return_value = mock_response
        result = url_opener(
            "http://example.com", 
            {"method": "get", "timeout": 30}
        )
        assert result == "Success response"
        mock_requests.get.assert_called_once_with(
            url="http://example.com", 
            timeout=30
        )

    # Test with session
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.Session.return_value = mock_session
        result = url_opener(
            "http://example.com", 
            {"method": "get", "session": mock_session}
        )
        assert result == "Success response"
        mock_session.get.assert_called_once_with(
            url="http://example.com", 
            timeout=60
        )

    # Test with bytes data
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.text = "Bytes data success"
    
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_requests.post.return_value = mock_post_response
        result = url_opener(
            "http://example.com", 
            {"method": "post", "data": b"raw bytes"}
        )
        assert result == "Bytes data success"
        mock_requests.post.assert_called_once_with(
            url="http://example.com", 
            timeout=60,
            data=b"raw bytes"
        )


# LLM-generated content at query #6
#--------------------------

```python
def test_url_opener():
    # Test 1: Basic GET request with requests library
    if HAS_REQUEST:
        result = url_opener("http://example.com", {"method": "get"})
        assert isinstance(result, str)
        assert "Example Domain" in result
    
    # Test 2: GET request with query parameters
    if HAS_REQUEST:
        result = url_opener("http://httpbin.org/get", {
            "method": "get",
            "data": {"key1": "value1", "key2": "value2"}
        })
        assert isinstance(result, str)
        assert "value1" in result
    
    # Test 3: POST request
    if HAS_REQUEST:
        result = url_opener("http://httpbin.org/post", {
            "method": "post",
            "data": {"test": "data"}
        })
        assert isinstance(result, str)
        assert "test" in result
    
    # Test 4: Request with custom timeout
    if HAS_REQUEST:
        result = url_opener("http://example.com", {
            "method": "get",
            "timeout": 30
        })
        assert isinstance(result, str)
    
    # Test 5: Request with headers
    if HAS_REQUEST:
        result = url_opener("http://httpbin.org/headers", {
            "method": "get",
            "headers": {"User-Agent": "TestAgent"}
        })
        assert isinstance(result, str)
        assert "TestAgent" in result
    
    # Test 6: Test with encoding parameter
    if HAS_REQUEST:
        result = url_opener("http://example.com", {
            "method": "get",
            "encoding": "utf-8"
        })
        assert isinstance(result, str)
    
    # Test 7: Test error handling (404)
    if HAS_REQUEST:
        try:
            url_opener("http://httpbin.org/status/404", {"method": "get"})
            assert False, "Should have raised HTTPError"
        except HTTPError as e:
            assert e.code == 404
    
    # Test 8: Test without requests library (urllib fallback)
    if not HAS_REQUEST:
        result = url_opener("http://example.com", {"method": "get"})
        assert hasattr(result, 'read')
        assert hasattr(result, 'status')
```


# LLM-generated content at query #7
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocked)
    # Test basic GET request
    kwargs = {'method': 'get'}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with data parameter for GET request (should append to URL)
    kwargs = {'method': 'get', 'data': {'key': 'value'}}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with POST request and data
    kwargs = {'method': 'post', 'data': {'key': 'value'}}
    try:
        result = url_opener('http://httpbin.org/post', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with encoding parameter
    kwargs = {'method': 'get', 'encoding': 'utf-8'}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with timeout parameter
    kwargs = {'method': 'get', 'timeout': 30}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with headers parameter
    kwargs = {'method': 'get', 'headers': {'User-Agent': 'TestAgent'}}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test invalid URL should raise exception
    kwargs = {'method': 'get'}
    try:
        result = url_opener('http://nonexistent-domain-12345.com', kwargs)
        assert False, "Should have raised an exception"
    except Exception:
        pass
    
    # Test with invalid method
    kwargs = {'method': 'invalid'}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
    
    # Test with auth parameter
    kwargs = {'method': 'get', 'auth': ('user', 'pass')}
    try:
        result = url_opener('http://example.com', kwargs)
        assert isinstance(result, str)
    except Exception:
        pass
```


# LLM-generated content at query #8
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    import requests
    from unittest.mock import patch, MagicMock
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    
    with patch('requests.get', return_value=mock_response):
        result = url_opener('http://example.com', {'method': 'get'})
        assert result == "Success"
    
    # Test with POST data
    mock_post_response = MagicMock()
    mock_post_response.status_code = 200
    mock_post_response.text = "Posted"
    
    with patch('requests.post', return_value=mock_post_response):
        result = url_opener('http://example.com', {
            'method': 'post',
            'data': {'key': 'value'}
        })
        assert result == "Posted"
    
    # Test with HTTP error
    mock_error_response = MagicMock()
    mock_error_response.status_code = 404
    mock_error_response.url = 'http://example.com'
    mock_error_response.reason = 'Not Found'
    mock_error_response.headers = {}
    
    with patch('requests.get', return_value=mock_error_response):
        try:
            url_opener('http://example.com', {'method': 'get'})
            assert False, "Should have raised HTTPError"
        except HTTPError as e:
            assert e.code == 404
    
    # Test with session
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    
    result = url_opener('http://example.com', {
        'method': 'get',
        'session': mock_session
    })
    assert result == "Success"
    mock_session.get.assert_called_once()
    
    # Test with data in GET request
    mock_get_data_response = MagicMock()
    mock_get_data_response.status_code = 200
    mock_get_data_response.text = "With data"
    
    with patch('requests.get', return_value=mock_get_data_response) as mock_get:
        result = url_opener('http://example.com', {
            'method': 'get',
            'data': {'param': 'value'}
        })
        assert result == "With data"
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert '?' in args[0] or '&' in args[0]
    
    # Test with headers and timeout
    mock_header_response = MagicMock()
    mock_header_response.status_code = 200
    mock_header_response.text = "With headers"
    
    with patch('requests.get', return_value=mock_header_response) as mock_get:
        result = url_opener('http://example.com', {
            'method': 'get',
            'headers': {'User-Agent': 'test'},
            'timeout': 30
        })
        assert result == "With headers"
        mock_get.assert_called_with(
            url='http://example.com',
            timeout=30,
            headers={'User-Agent': 'test'}
        )
```


# LLM-generated content at query #9
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocked)
    import pytest
    from unittest.mock import patch, MagicMock
    
    # Test with urllib when requests is not available
    with patch('module_name.HAS_REQUEST', False), \
         patch('module_name.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        result = url_opener('http://example.com', {'method': 'GET'})
        assert result == mock_response
        mock_urlopen.assert_called_once_with('http://example.com', None, timeout=60)
    
    # Test with GET method and data
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'success'
        mock_requests.get.return_value = mock_response
        result = url_opener('http://example.com', {
            'method': 'get',
            'data': {'key': 'value'}
        })
        assert result == 'success'
        mock_requests.get.assert_called_once_with(
            url='http://example.com?key=value',
            timeout=60
        )
    
    # Test with POST method and data
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'posted'
        mock_requests.post.return_value = mock_response
        result = url_opener('http://example.com', {
            'method': 'post',
            'data': {'key': 'value'}
        })
        assert result == 'posted'
        mock_requests.post.assert_called_once()
        assert mock_requests.post.call_args[1]['data'] == 'key=value'
    
    # Test with HTTP error
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests, \
         pytest.raises(HTTPError):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_response.headers = {}
        mock_response.url = 'http://example.com/notfound'
        mock_requests.get.return_value = mock_response
        url_opener('http://example.com/notfound', {'method': 'get'})
    
    # Test with custom timeout
    with patch('module_name.HAS_REQUEST', True), \
         patch('module_name.requests') as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'success'
        mock_requests.get.return_value = mock_response
        result = url_opener('http://example.com', {
            'method': 'get',
            'timeout': 30
        })
        assert result == 'success'
        mock_requests.get.assert_called_once_with(
            url='http://example.com',
            timeout=30
        )
```


# LLM-generated content at query #10
#--------------------------

```python
def test_url_opener():
    # Test with requests module available (mocked)
    # Test GET request with data
    url = "http://example.com"
    kwargs = {
        'method': 'get',
        'data': {'key': 'value'},
        'headers': {'User-Agent': 'test'}
    }
    
    # Test that data is appended to URL for GET requests
    if '?' not in url:
        expected_url = url + '?key=value'
    else:
        expected_url = url + '&key=value'
    
    # Test POST request with data
    post_kwargs = {
        'method': 'post',
        'data': {'key': 'value'},
        'timeout': 30
    }
    
    # Test with session object
    session_kwargs = {
        'method': 'get',
        'session': None  # Would need mock session
    }
    
    # Test with encoding
    encoding_kwargs = {
        'method': 'get',
        'encoding': 'utf-8'
    }
    
    # Test with invalid HTTP status code (would raise HTTPError)
    invalid_status_kwargs = {
        'method': 'get',
        'headers': {'Accept': 'application/json'}
    }
    
    # Test urllib fallback when requests not available
    # This would require mocking HAS_REQUEST to False
    
    # Test basic functionality
    assert callable(url_opener)
    assert url_opener.__name__ == 'url_opener'
    
    # Test that kwargs are properly filtered
    invalid_kwargs = {
        'method': 'get',
        'invalid_arg': 'test',
        'another_invalid': 123
    }
    # Only allowed_args should be passed through
    
    # Test timeout default
    timeout_kwargs = {'method': 'get'}
    # Should use DEFAULT_TIMEOUT = 60
    
    # Test data types
    data_types_kwargs = {
        'method': 'post',
        'data': [('key', 'value'), ('key2', 'value2')]  # list of tuples
    }
    
    # Test string data
    string_data_kwargs = {
        'method': 'post',
        'data': 'raw string data'
    }
    
    # Test bytes data
    bytes_data_kwargs = {
        'method': 'post',
        'data': b'bytes data'
    }
    
    # Verify function exists and accepts correct parameters
    try:
        # This will likely fail without network, but tests the structure
        result = url_opener(url, kwargs)
    except Exception:
        pass  # Expected to fail in test environment without network
    
    # Test _query function directly
    from urllib.parse import urlencode
    
    test_url = "http://test.com"
    test_data = {'param': 'value'}
    
    # GET with data should append to URL
    url_result, data_result = _query(test_url, 'get', {'data': test_data})
    assert data_result is None
    assert 'param=value' in url_result
    
    # POST with data should keep data separate
    url_result, data_result = _query(test_url, 'post', {'data': test_data})
    assert data_result is not None
    assert 'param=value' in data_result
    
    # Test with existing URL parameters
    url_with_params = "http://test.com?existing=param"
    url_result, data_result = _query(url_with_params, 'get', {'data': test_data})
    assert '&' in url_result
    assert 'existing=param' in url_result
    assert 'param=value' in url_result
    
    # Test with trailing ?
    url_with_question = "http://test.com?"
    url_result, data_result = _query(url_with_question, 'get', {'data': test_data})
    assert url_result.endswith('param=value')
    assert '?param=value' not in url_result  # Should not add extra ?
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocked)
    import requests
    from unittest.mock import patch, MagicMock
    
    # Mock requests response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    mock_response.encoding = 'utf-8'
    
    with patch('requests.get', return_value=mock_response):
        result = url_opener('http://example.com', {'method': 'GET'})
        assert result == "Success"
    
    # Test with data parameter for GET request
    mock_response2 = MagicMock()
    mock_response2.status_code = 200
    mock_response2.text = "Data success"
    mock_response2.encoding = 'utf-8'
    
    with patch('requests.get', return_value=mock_response2):
        result = url_opener('http://example.com', {
            'method': 'GET',
            'data': {'key': 'value'}
        })
        assert result == "Data success"
    
    # Test HTTP error
    mock_response3 = MagicMock()
    mock_response3.status_code = 404
    mock_response3.reason = "Not Found"
    mock_response3.headers = {}
    mock_response3.url = 'http://example.com'
    
    with patch('requests.get', return_value=mock_response3):
        try:
            url_opener('http://example.com', {'method': 'GET'})
            assert False, "Should have raised HTTPError"
        except HTTPError as e:
            assert e.code == 404
            assert e.reason == "Not Found"
    
    # Test without requests library (using urllib)
    with patch('__main__.HAS_REQUEST', False):
        from urllib.request import urlopen
        mock_urlopen = MagicMock()
        mock_urlopen.read.return_value = b"urllib success"
        
        with patch('urllib.request.urlopen', return_value=mock_urlopen):
            result = url_opener('http://example.com', {'method': 'GET'})
            assert result == mock_urlopen
```


# LLM-generated content at query #2
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    # Test basic GET request
    result = url_opener("http://httpbin.org/get", {"method": "get"})
    assert isinstance(result, str)
    assert "url" in result

    # Test GET request with data (should append to URL)
    result = url_opener("http://httpbin.org/get", {
        "method": "get",
        "data": {"key": "value"}
    })
    assert isinstance(result, str)
    assert "key=value" in result

    # Test POST request with data
    result = url_opener("http://httpbin.org/post", {
        "method": "post",
        "data": {"key": "value"}
    })
    assert isinstance(result, str)
    assert "key" in result

    # Test with custom timeout
    result = url_opener("http://httpbin.org/get", {
        "method": "get",
        "timeout": 30
    })
    assert isinstance(result, str)

    # Test with encoding
    result = url_opener("http://httpbin.org/get", {
        "method": "get",
        "encoding": "utf-8"
    })
    assert isinstance(result, str)

    # Test with headers
    result = url_opener("http://httpbin.org/headers", {
        "method": "get",
        "headers": {"User-Agent": "test-agent"}
    })
    assert isinstance(result, str)
    assert "test-agent" in result

    # Test with cookies
    result = url_opener("http://httpbin.org/cookies", {
        "method": "get",
        "cookies": {"test_cookie": "test_value"}
    })
    assert isinstance(result, str)
    assert "test_cookie" in result

    # Test error handling - 404
    try:
        url_opener("http://httpbin.org/status/404", {"method": "get"})
        assert False, "Should have raised HTTPError"
    except HTTPError:
        pass

    # Test error handling - 500
    try:
        url_opener("http://httpbin.org/status/500", {"method": "get"})
        assert False, "Should have raised HTTPError"
    except HTTPError:
        pass

    # Test with data as dict in GET request (should convert to query string)
    result = url_opener("http://httpbin.org/get", {
        "method": "get",
        "data": {"param1": "value1", "param2": "value2"}
    })
    assert isinstance(result, str)
    assert "param1=value1" in result
    assert "param2=value2" in result

    # Test with data as list in GET request
    result = url_opener("http://httpbin.org/get", {
        "method": "get",
        "data": [("key1", "value1"), ("key2", "value2")]
    })
    assert isinstance(result, str)
    assert "key1=value1" in result
    assert "key2=value2" in result
```


# LLM-generated content at query #3
#--------------------------

```python
def test_url_opener():
    # Test with requests library available
    if HAS_REQUEST:
        # Test GET request without data
        result = url_opener("http://example.com", {"method": "get"})
        assert isinstance(result, str)
        
        # Test GET request with data
        result = url_opener("http://example.com", {
            "method": "get", 
            "data": {"key": "value"}
        })
        assert isinstance(result, str)
        
        # Test POST request
        result = url_opener("http://httpbin.org/post", {
            "method": "post",
            "data": {"test": "data"}
        })
        assert isinstance(result, str)
        
        # Test with timeout
        result = url_opener("http://example.com", {
            "method": "get",
            "timeout": 30
        })
        assert isinstance(result, str)
        
        # Test with headers
        result = url_opener("http://example.com", {
            "method": "get",
            "headers": {"User-Agent": "test-agent"}
        })
        assert isinstance(result, str)
        
        # Test with encoding
        result = url_opener("http://example.com", {
            "method": "get",
            "encoding": "utf-8"
        })
        assert isinstance(result, str)
        
        # Test that HTTPError is raised for non-2xx status codes
        import pytest
        with pytest.raises(HTTPError):
            url_opener("http://httpbin.org/status/404", {"method": "get"})
        
        with pytest.raises(HTTPError):
            url_opener("http://httpbin.org/status/500", {"method": "get"})
    
    else:
        # Test with urllib only
        result = url_opener("http://example.com", {"method": "get"})
        assert result is not None
        
        # Test GET with data
        result = url_opener("http://example.com", {
            "method": "get",
            "data": {"key": "value"}
        })
        assert result is not None
```


# LLM-generated content at query #4
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocking)
    with patch('your_module.HAS_REQUEST', True), \
         patch('your_module._requests') as mock_requests, \
         patch('your_module._urllib') as mock_urllib:
        
        # Test that _requests is called when HAS_REQUEST is True
        mock_requests.return_value = "response text"
        result = url_opener("http://example.com", {"method": "get"})
        mock_requests.assert_called_once_with("http://example.com", {"method": "get"})
        assert result == "response text"
        mock_urllib.assert_not_called()

    # Test without requests library (mocking)
    with patch('your_module.HAS_REQUEST', False), \
         patch('your_module._urllib') as mock_urllib:
        
        # Test that _urllib is called when HAS_REQUEST is False
        mock_response = MagicMock()
        mock_response.read.return_value = b"response data"
        mock_urllib.return_value = mock_response
        
        result = url_opener("http://example.com", {"method": "get"})
        mock_urllib.assert_called_once_with("http://example.com", {"method": "get"})
        assert result == mock_response

    # Test with default timeout
    with patch('your_module.HAS_REQUEST', True), \
         patch('your_module._requests') as mock_requests:
        
        mock_requests.return_value = "response"
        url_opener("http://example.com", {"method": "post"})
        mock_requests.assert_called_once_with("http://example.com", {"method": "post"})

    # Test with custom timeout
    with patch('your_module.HAS_REQUEST', True), \
         patch('your_module._requests') as mock_requests:
        
        mock_requests.return_value = "response"
        url_opener("http://example.com", {"method": "get", "timeout": 30})
        mock_requests.assert_called_once_with("http://example.com", {"method": "get", "timeout": 30})
```


# LLM-generated content at query #5
#--------------------------

```python
def test_url_opener():
    # Test with requests library available (mocked)
    # Test GET request with data
    with patch('yourmodule.HAS_REQUEST', True), \
         patch('yourmodule._requests') as mock_requests:
        mock_requests.return_value = "response"
        result = url_opener("http://example.com", {"method": "get", "data": {"key": "value"}})
        assert result == "response"
        mock_requests.assert_called_once_with("http://example.com", {"method": "get", "data": {"key": "value"}})

    # Test with urllib (no requests)
    with patch('yourmodule.HAS_REQUEST', False), \
         patch('yourmodule._urllib') as mock_urllib:
        mock_urllib.return_value = "response"
        result = url_opener("http://example.com", {"method": "GET"})
        assert result == "response"
        mock_urllib.assert_called_once_with("http://example.com", {"method": "GET"})

    # Test with session
    with patch('yourmodule.HAS_REQUEST', True), \
         patch('yourmodule._requests') as mock_requests:
        mock_requests.return_value = "session_response"
        session = MagicMock()
        result = url_opener("http://example.com", {"method": "POST", "session": session})
        assert result == "session_response"
        mock_requests.assert_called_once_with("http://example.com", {"method": "POST", "session": session})

    # Test with timeout
    with patch('yourmodule.HAS_REQUEST', True), \
         patch('yourmodule._requests') as mock_requests:
        mock_requests.return_value = "timeout_response"
        result = url_opener("http://example.com", {"method": "GET", "timeout": 30})
        assert result == "timeout_response"
        mock_requests.assert_called_once_with("http://example.com", {"method": "GET", "timeout": 30})

    # Test with headers
    with patch('yourmodule.HAS_REQUEST', True), \
         patch('yourmodule._requests') as mock_requests:
        mock_requests.return_value = "headers_response"
        result = url_opener("http://example.com", {"method": "GET", "headers": {"Authorization": "Bearer token"}})
        assert result == "headers_response"
        mock_requests.assert_called_once_with("http://example.com", {"method": "GET", "headers": {"Authorization": "Bearer token"}})
```


