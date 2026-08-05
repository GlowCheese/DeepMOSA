# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'post'
    var_1 = {var_0: var_0, var_0: var_0}
    module_0.url_opener(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'data'
    var_1 = 'get'
    var_2 = {var_1: var_1}
    var_3 = {var_1: var_1, var_0: var_2}
    module_0.url_opener(var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = {var_1: var_0}
    module_0.url_opener(var_1, var_2)

def test_case_4():
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
def test_case_5():
    var_0 = 'http://example.com'
    var_1 = 'method'
    var_2 = 'data'
    var_3 = 'get'
    var_4 = {var_0: var_0}
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.url_opener(var_0, var_5)
    assert var_6 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_7 = 'post'
    var_8 = {var_0: var_1}
    var_9 = {var_1: var_7, var_2: var_8}
    module_0.url_opener(var_0, var_9)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'bp;F\\\\;}hq5oX.?6o '
    var_1 = 'kLr&b0W/nhy_'
    var_2 = 'data'
    var_3 = 'ge]'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    module_0.url_opener(var_0, var_7)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'lmwLz\n#ag.o#(RpdB?'
    var_1 = 'kLr&b@XP0W/nhy_'
    var_2 = 'data'
    var_3 = 'ge]'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_3, var_2: var_6}
    module_0.url_opener(var_0, var_7)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'http://example.com'
    var_1 = 'param'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'http://example.com'
    var_5 = module_0.url_opener(var_4, var_3)
    assert var_5 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')
    var_6 = 'http://example.com'
    var_7 = 'encoding'
    var_8 = 'get'
    var_9 = 'utf-8'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = module_0.url_opener(var_6, var_10)
    assert var_11 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    var_12 = 'http://example.com/notfound'
    var_13 = 'method'
    var_14 = 'get'
    var_15 = {var_13: var_14}
    module_0.url_opener(var_12, var_15)