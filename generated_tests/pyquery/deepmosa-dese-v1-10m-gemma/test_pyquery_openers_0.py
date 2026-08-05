# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import http.cookies as module_1
import urllib.parse as module_2
import email._encoded_words as module_3
import urllib.error as module_4

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
    var_1 = module_1.Morsel()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'http.cookies.Morsel'
    assert len(var_1) == 9
    assert f'{type(module_1.Morsel.key).__module__}.{type(module_1.Morsel.key).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.value).__module__}.{type(module_1.Morsel.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.coded_value).__module__}.{type(module_1.Morsel.coded_value).__qualname__}' == 'builtins.property'
    var_2 = module_0.url_opener(var_0, var_1)
    assert var_2 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_3():
    var_0 = 'http://example.com'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_2: var_2, var_1: var_3}
    var_5 = module_0.url_opener(var_0, var_4)
    assert var_5 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'http://exaple.com'
    var_1 = module_1.Morsel()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'http.cookies.Morsel'
    assert len(var_1) == 9
    assert f'{type(module_1.Morsel.key).__module__}.{type(module_1.Morsel.key).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.value).__module__}.{type(module_1.Morsel.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Morsel.coded_value).__module__}.{type(module_1.Morsel.coded_value).__qualname__}' == 'builtins.property'
    module_0.url_opener(var_0, var_1)

def test_case_5():
    var_0 = 'http://example.com?a=b&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_2.urlencode(var_7)
    assert var_8 == 'k=v'
    assert module_2.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_2.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_2.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_2.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_2.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_2.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_2.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_2.MAX_CACHE_SIZE == 20
    var_9 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_6():
    var_0 = 'http://example.com'
    var_1 = 'POST'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_2.urlencode(var_7)
    assert var_8 == 'key=value'
    assert module_2.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_2.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_2.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_2.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_2.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_2.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_2.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_2.MAX_CACHE_SIZE == 20
    var_9 = 'utf-8'
    var_10 = module_3.encode(var_9)
    assert var_10 == '=?utf-8?q?utf-8?='
    assert module_3.ascii_letters == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert module_3.digits == '0123456789'
    var_11 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_7():
    var_0 = 'c'
    var_1 = 'd'
    var_2 = {var_0: var_1}
    var_3 = module_2.urlencode(var_2)
    assert var_3 == 'c=d'
    assert module_2.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_2.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_2.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_2.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_2.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_2.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_2.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_2.MAX_CACHE_SIZE == 20
    var_4 = module_0._query(var_3, var_3, var_3)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_8():
    var_0 = 'http://example.com'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = module_2.urlencode(var_7)
    assert var_8 == 'key=value'
    assert module_2.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_2.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_2.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_2.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_2.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_2.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_2.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_2.MAX_CACHE_SIZE == 20
    var_9 = 'http://example.com?'
    var_10 = var_9 + var_8
    var_11 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_9():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'post'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    with pytest.raises(module_4.HTTPError):
        module_0._requests(var_5, var_4)
    assert var_6 == 'success'

def test_case_10():
    var_0 = 'utf-8'
    var_1 = None
    var_2 = module_0._query(var_1, var_1, var_0)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_11():
    var_0 = 'method'
    var_1 = 'data'
    var_2 = 'post'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = 'http://example.com'
    with pytest.raises(module_4.HTTPError):
        module_0._requests(var_7, var_6)