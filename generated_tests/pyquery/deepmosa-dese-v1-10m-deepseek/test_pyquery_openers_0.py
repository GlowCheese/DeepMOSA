# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import http.cookies as module_1
import requests.sessions as module_2
import urllib.error as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.Morsel()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'http.cookies.Morsel'
    assert len(var_0) == 9
    assert f'{type(module_1.Morsel.key).__module__}.{type(module_1.Morsel.key).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.value).__module__}.{type(module_1.Morsel.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.coded_value).__module__}.{type(module_1.Morsel.coded_value).__qualname__}' == 'builtins.property'
    module_0.url_opener(var_0, var_0)

def test_case_2():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = {var_1: var_1}
    module_0._requests(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'http://xample.com'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = {var_1: var_1}
    var_4 = None
    var_5 = {var_0: var_2, var_1: var_3, var_2: var_0, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    module_0.url_opener(var_6, var_3)

def test_case_5():
    var_0 = module_2.Session()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'requests.sessions.Session'
    assert f'{type(var_0.headers).__module__}.{type(var_0.headers).__qualname__}' == 'requests.structures.CaseInsensitiveDict'
    assert len(var_0.headers) == 4
    assert var_0.auth is None
    assert var_0.proxies == {}
    assert var_0.hooks == {'response': []}
    assert var_0.params == {}
    assert var_0.stream is False
    assert var_0.verify is True
    assert var_0.cert is None
    assert var_0.max_redirects == 30
    assert var_0.trust_env is True
    assert f'{type(var_0.cookies).__module__}.{type(var_0.cookies).__qualname__}' == 'requests.cookies.RequestsCookieJar'
    assert len(var_0.cookies) == 0
    assert f'{type(var_0.adapters).__module__}.{type(var_0.adapters).__qualname__}' == 'collections.OrderedDict'
    assert len(var_0.adapters) == 2
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.TYPE_CHECKING is False
    assert module_2.DEFAULT_REDIRECT_LIMIT == 30
    assert module_2.REDIRECT_STATI == (301, 302, 303, 307, 308)
    assert f'{type(module_2.codes).__module__}.{type(module_2.codes).__qualname__}' == 'requests.structures.LookupDict'
    assert len(module_2.codes) == 0
    assert module_2.DEFAULT_PORTS == {'http': 80, 'https': 443}
    var_1 = 'http://example.com'
    var_2 = 'method'
    var_3 = 'encoding'
    var_4 = 'session'
    var_5 = 'get'
    var_6 = 'utf-8'
    var_7 = {var_2: var_5, var_3: var_6, var_4: var_0}
    var_8 = module_0._requests(var_1, var_7)
    assert var_8 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_6():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_7():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'session'
    var_4 = 'post'
    var_5 = 'utf-8'
    var_6 = None
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    with pytest.raises(module_3.HTTPError):
        module_0._requests(var_0, var_7)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'get'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://httpbin.org/status/404'
    module_0.url_opener(var_5, var_4)

def test_case_9():
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
    assert var_10 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_10():
    var_0 = 'http://example.com'
    var_1 = 123
    var_2 = 'data'
    var_3 = 'some_data'
    var_4 = {var_2: var_3}
    var_5 = module_0._query(var_0, var_1, var_4)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'http://xample.com'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = {var_0: var_2, var_1: var_3, var_2: var_0, var_1: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    module_0.url_opener(var_6, var_3)

def test_case_12():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0._urllib(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'http.client.HTTPResponse'
    assert f'{type(var_4.fp).__module__}.{type(var_4.fp).__qualname__}' == '_io.BufferedReader'
    assert var_4.debuglevel == 0
    assert f'{type(var_4.headers).__module__}.{type(var_4.headers).__qualname__}' == 'http.client.HTTPMessage'
    assert len(var_4.headers) == 11
    assert var_4.msg == 'OK'
    assert var_4.version == 11
    assert var_4.status == 200
    assert var_4.reason == 'OK'
    assert var_4.chunked is True
    assert var_4.chunk_left is None
    assert var_4.length is None
    assert var_4.will_close is True
    assert var_4.code == 200
    assert var_4.url == 'http://example.com'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')