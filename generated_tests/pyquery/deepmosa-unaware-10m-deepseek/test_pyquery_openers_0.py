# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import urllib.request as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 0
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_5 = '7TmB;Udc'
    var_6 = 'data'
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_1: var_2, var_6: var_11}
    module_0.url_opener(var_5, var_12)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'http://httpbin.org/get'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_5 = 'data'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_2, var_5: var_8}
    var_10 = module_0.url_opener(var_0, var_9)
    var_11 = 'http://httpbin.org/post'
    var_12 = 'post'
    var_13 = {var_6: var_7}
    var_14 = {var_1: var_12, var_5: var_13}
    var_15 = module_0.url_opener(var_11, var_14)
    var_16 = 'http://httpbin.org/headers'
    var_17 = 'headers'
    var_18 = 'X-Test'
    var_19 = 'test-value'
    var_20 = {var_18: var_19}
    var_21 = {var_1: var_2, var_17: var_20}
    var_22 = module_0.url_opener(var_16, var_21)
    var_23 = 'http://httpbin.org/delay/1'
    var_24 = 'timeout'
    var_25 = 5
    var_26 = {var_1: var_2, var_24: var_25}
    var_27 = module_0.url_opener(var_23, var_26)
    var_28 = 'session'
    var_29 = {var_1: var_2, var_28: var_4}
    module_0.url_opener(var_0, var_29)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'http://httpbin.org/get'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_4.clear()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == 'success'
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_5 = 'http://example.com'
    var_6 = 'method'
    var_7 = 'data'
    var_8 = 'post'
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = {var_6: var_8, var_7: var_11}
    module_0.url_opener(var_5, var_12)
    assert var_13 == 'success'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_5 = 'http://httpbin.org/get'
    var_6 = 'data'
    var_7 = 'key1'
    var_8 = 'key2'
    var_9 = 'value1'
    var_10 = 'value2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_1: var_2, var_6: var_11}
    var_13 = module_0.url_opener(var_5, var_12)
    var_14 = 'http://httpbin.org/post'
    var_15 = 'post'
    var_16 = 'test'
    var_17 = {var_16: var_6}
    var_18 = {var_1: var_15, var_6: var_17}
    var_19 = module_0.url_opener(var_14, var_18)
    var_20 = 'timeout'
    var_21 = 30
    var_22 = {var_1: var_2, var_20: var_21}
    var_23 = module_0.url_opener(var_0, var_22)
    assert var_23 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_24 = 'http://httpbin.org/headers'
    var_25 = 'headers'
    var_26 = 'User-Agent'
    var_27 = 'test-agent'
    var_28 = {var_26: var_27}
    var_29 = {var_1: var_2, var_25: var_28}
    var_30 = module_0.url_opener(var_24, var_29)
    var_31 = 'encoding'
    var_32 = 'utf-8'
    var_33 = {var_1: var_2, var_31: var_32}
    var_34 = module_0.url_opener(var_0, var_33)
    assert var_34 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_35 = 'http://httpbin.org/status/404'
    var_36 = 'method'
    var_37 = 'get'
    var_38 = {var_36: var_37}
    module_0.url_opener(var_35, var_38)