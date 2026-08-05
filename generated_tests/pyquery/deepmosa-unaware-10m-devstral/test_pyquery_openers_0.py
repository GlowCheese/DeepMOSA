# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import email._encoded_words as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'get'
    var_1 = {var_0: var_0}
    module_0.url_opener(var_0, var_1)

def test_case_2():
    var_0 = 'http://example.com'
    var_1 = {var_0: var_0}
    var_2 = 'http://example.com'
    var_3 = 'data'
    var_4 = 'post'
    var_5 = {var_2: var_4, var_3: var_1}
    var_6 = module_0.url_opener(var_2, var_5)
    assert var_6 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_7 = module_1.encode(var_6)
    assert var_7 == '=?utf-8?b?PCFkb2N0eXBlIGh0bWw+PGh0bWwgbGFuZz0iZW4iPjxoZWFkPjx0aXRsZT5FeGFtcGxlIERvbWFpbjwvdGl0bGU+PGxpbmsgcmVsPSJpY29uIiBocmVmPSJkYXRhOiwiPjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MSI+PHN0eWxlPmJvZHl7YmFja2dyb3VuZDojZWVlO3dpZHRoOjYwdnc7bWFyZ2luOjE1dmggYXV0bztmb250LWZhbWlseTpzeXN0ZW0tdWksc2Fucy1zZXJpZn1oMXtmb250LXNpemU6MS41ZW19ZGl2e29wYWNpdHk6MC44fWE6bGluayxhOnZpc2l0ZWR7Y29sb3I6IzM0OH08L3N0eWxlPjwvaGVhZD48Ym9keT48ZGl2PjxoMT5FeGFtcGxlIERvbWFpbjwvaDE+PHA+VGhpcyBkb21haW4gaXMgZm9yIHVzZSBpbiBkb2N1bWVudGF0aW9uIGV4YW1wbGVzIHdpdGhvdXQgbmVlZGluZyBwZXJtaXNzaW9uLiBBdm9pZCB1c2UgaW4gb3BlcmF0aW9ucy48L3A+PHA+PGEgaHJlZj0iaHR0cHM6Ly9pYW5hLm9yZy9kb21haW5zL2V4YW1wbGUiPkxlYXJuIG1vcmU8L2E+PC9wPjwvZGl2PjwvYm9keT48L2h0bWw+Cg==?='
    assert module_1.ascii_letters == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert module_1.digits == '0123456789'

def test_case_3():
    var_0 = 'http://example.com'
    var_1 = {var_0: var_0}
    var_2 = 'post'
    var_3 = {var_0: var_2, var_2: var_1}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'get'
    var_1 = 'http://example.com'
    var_2 = '|E|M_k\x0ck2M[hA'
    var_3 = {var_2: var_0}
    var_4 = module_0.url_opener(var_1, var_3)
    assert var_4 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_5 = 'method'
    var_6 = 'data'
    var_7 = 'post'
    var_8 = {var_5: var_7, var_6: var_3}
    module_0.url_opener(var_0, var_8)
    assert var_9 == 'test content'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'method'
    var_1 = 'http://example.com'
    var_2 = '|E|M_k\x0ck2M[hA'
    var_3 = {var_2: var_0}
    var_4 = module_0.url_opener(var_1, var_3)
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
    var_9 = 'value'
    var_10 = {var_9: var_9}
    var_11 = {var_6: var_8, var_7: var_10}
    module_0.url_opener(var_5, var_11)
    assert var_12 == 'test content'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'method'
    var_1 = {var_0: var_0}
    var_2 = '+6fnqY@zWzq?B'
    var_3 = 'method'
    var_4 = 'data'
    var_5 = 'get'
    var_6 = {var_3: var_5, var_4: var_1}
    module_0.url_opener(var_2, var_6)