# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import http.cookies as module_1

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
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = ''
    var_5 = {var_4: var_1}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = module_0.url_opener(var_0, var_6)
    assert var_7 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_3():
    var_0 = 'http://example.com'
    var_1 = {var_0: var_0}
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'method'
    var_1 = 'key'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = 'post'
    var_5 = {var_0: var_3, var_0: var_4}
    module_0.url_opener(var_4, var_5)
    assert var_6 == 'Created'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'timeout'
    var_1 = 'headers'
    var_2 = 10
    var_3 = 'User-Agent'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://test.com'
    module_0.url_opener(var_7, var_6)
    assert var_8 == '<html>success</html>'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'http://test.com?existing=true'
    var_1 = 'data'
    var_2 = 'get'
    var_3 = 'new'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_2, var_1: var_5}
    module_0.url_opener(var_0, var_6)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    var_4 = 'method'
    var_5 = 'data'
    var_6 = 'post'
    var_7 = {var_4: var_6, var_5: var_2}
    module_0.url_opener(var_3, var_7)
    assert var_8 == 'Created'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'session'
    var_1 = '123'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com'
    module_0.url_opener(var_3, var_2)

def test_case_9():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'utf-8'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')