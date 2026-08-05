# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import posixpath as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'get'
    var_1 = {var_0: var_0}
    module_0.url_opener(var_0, var_1)
    assert var_2 == 'test content'

def test_case_2():
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
def test_case_3():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'get'
    var_3 = {var_1: var_2}
    var_4 = module_0.url_opener(var_0, var_3)
    assert var_4 == b'test content'
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
    assert var_13 == b'test content'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'get'
    var_1 = '#]ZEm?\n70['
    var_2 = 'data'
    var_3 = {var_1: var_0}
    var_4 = {var_0: var_0, var_2: var_3}
    module_0.url_opener(var_1, var_4)
    assert var_5 == 'test content'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'http://example.BoK'
    var_1 = ' hoPa'
    var_2 = module_1.lexists(var_1)
    assert var_2 is False
    assert module_1.curdir == '.'
    assert module_1.pardir == '..'
    assert module_1.extsep == '.'
    assert module_1.sep == '/'
    assert module_1.pathsep == ':'
    assert module_1.defpath == '/bin:/usr/bin'
    assert module_1.altsep is None
    assert module_1.devnull == '/dev/null'
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_1.supports_unicode_filenames is False
    var_3 = 'http://example.com'
    var_4 = 'data'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_5, var_4: var_7}
    var_9 = module_0.url_opener(var_3, var_8)
    assert var_9 == 'test content'
    assert var_9 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_10 = 'method'
    var_11 = 'get'
    var_12 = {var_10: var_11}
    module_0.url_opener(var_11, var_12)
    assert var_13 == b'test content'

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
    var_5 = 'http://example.com'
    var_6 = 'mto'
    var_7 = 'get'
    var_8 = {var_6: var_7}
    var_9 = module_0.url_opener(var_5, var_8)
    assert var_9 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_10 = 'http://example.com'
    var_11 = 'method'
    var_12 = 'get'
    var_13 = {var_11: var_12}
    var_14 = module_0.url_opener(var_10, var_13)
    assert var_14 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_15 = 'http://example.com'
    var_16 = module_0.url_opener(var_15, var_13)
    assert var_16 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_17 = 'http://example.com'
    var_18 = 'method'
    var_19 = 'encoding'
    var_20 = 'get'
    var_21 = 'latin-1'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = module_0.url_opener(var_17, var_22)
    assert var_23 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'get'
    var_1 = 'll\tqZ^Z;s<T?'
    var_2 = 'data'
    var_3 = {var_1: var_0}
    var_4 = {var_0: var_0, var_2: var_3}
    module_0.url_opener(var_1, var_4)
    assert var_5 == 'test content'