# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import requests.utils as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.default_headers()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'requests.structures.CaseInsensitiveDict'
    assert len(var_0) == 4
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert f'{type(module_1.HEADER_VALIDATORS).__module__}.{type(module_1.HEADER_VALIDATORS).__qualname__}' == 'builtins.dict'
    assert len(module_1.HEADER_VALIDATORS) == 2
    assert f'{type(module_1.integer_types).__module__}.{type(module_1.integer_types).__qualname__}' == 'builtins.tuple'
    assert len(module_1.integer_types) == 1
    assert module_1.is_urllib3_1 is False
    assert module_1.NETRC_FILES == ('.netrc', '_netrc')
    assert module_1.DEFAULT_CA_BUNDLE_PATH == '/usr/local/lib/python3.10/site-packages/certifi/cacert.pem'
    assert module_1.DEFAULT_PORTS == {'http': 80, 'https': 443}
    assert module_1.DEFAULT_ACCEPT_ENCODING == 'gzip, deflate'
    assert f'{type(module_1.UNRESERVED_SET).__module__}.{type(module_1.UNRESERVED_SET).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.UNRESERVED_SET) == 66
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "iP)''=L\tf\n"
    var_1 = 10
    var_2 = 'data'
    var_3 = 'get'
    var_4 = ''
    var_5 = {var_4: var_1}
    var_6 = {var_0: var_3, var_2: var_5}
    module_0.url_opener(var_3, var_6)
    assert var_7 == '<html>success</html>'

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
    var_0 = 'key'
    var_1 = 'val'
    var_2 = 'http://example.com'
    var_3 = 'method'
    var_4 = 'data'
    var_5 = 'get'
    var_6 = 'a'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = module_0.url_opener(var_2, var_9)
    assert var_10 == '<html>success</html>'
    assert var_10 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_11 = 'post'
    var_12 = {var_0: var_1}
    var_13 = {var_3: var_11, var_4: var_12}
    module_0.url_opener(var_2, var_13)
    assert var_14 == 'created'

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    module_0.url_opener(var_0, var_14)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'method'
    var_1 = 'session'
    var_2 = '123'
    var_3 = {var_1: var_2}
    module_0.url_opener(var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'http://example.com'
    var_1 = {}
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_3 = 'data'
    var_4 = 'timeout'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 10
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = module_0.url_opener(var_0, var_9)
    assert var_10 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_11 = 'encoding'
    var_12 = 'utf-8'
    var_13 = {var_11: var_12}
    var_14 = module_0.url_opener(var_0, var_13)
    assert var_14 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_15 = 'http://example.com'
    var_16 = {}
    var_17 = module_0.url_opener(var_15, var_16)
    assert var_17 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_18 = 'http://example.com'
    var_19 = {}
    var_20 = module_0.url_opener(var_18, var_19)
    assert var_20 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_21 = 'method'
    var_22 = 'data'
    var_23 = 'post'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = {var_24: var_25}
    var_27 = {var_21: var_23, var_22: var_26}
    module_0.url_opener(var_18, var_27)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = "iP)''=L\tf\n"
    var_1 = 10
    var_2 = 'HL8|>!_yHy\x0b?*'
    var_3 = 'data'
    var_4 = 'get'
    var_5 = 'r'
    var_6 = {var_5: var_1}
    var_7 = {var_0: var_4, var_3: var_6}
    module_0.url_opener(var_2, var_7)
    assert var_8 == '<html>success</html>'