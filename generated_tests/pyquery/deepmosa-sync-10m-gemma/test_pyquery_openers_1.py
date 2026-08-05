# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import email._encoded_words as module_1
import urllib.error as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = {}
    module_0.url_opener(var_0, var_0)

def test_case_2():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'get'
    var_1 = 'data'
    var_2 = 'key'
    var_3 = 'val'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0._query(var_2, var_0, var_5)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    module_0.url_opener(var_6, var_6)

def test_case_4():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'raw_string'
    var_4 = {var_2: var_3}
    var_5 = 'http://example.com'
    var_6 = 'utf-8'
    var_7 = module_1.encode(var_6)
    assert var_7 == '=?utf-8?q?utf-8?='
    assert module_1.ascii_letters == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert module_1.digits == '0123456789'
    var_8 = module_0._query(var_0, var_1, var_4)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_9 = bool(var_8 == (var_5, var_7))

def test_case_5():
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
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

def test_case_6():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'encoding'
    var_3 = 'get'
    var_4 = 'latin-1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0._requests(var_0, var_5)
    assert var_6 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'params'
    var_1 = 'headers'
    var_2 = [var_0, var_0, var_1]
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_4}
    var_6 = 10
    var_7 = {var_0: var_5, var_3: var_2, var_1: var_6}
    var_8 = 'http://example.com'
    module_0._requests(var_8, var_7)
    assert var_9 == '<html>content</html>'

def test_case_8():
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
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_10 = bool(var_9 == (var_7, var_8))
    assert var_10 is True

def test_case_9():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com/error'
    with pytest.raises(module_2.HTTPError):
        module_0._requests(var_3, var_2)