# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyquery.openers as module_0
import _locale as module_1
import urllib.error as module_2
import urllib.parse as module_3
import email._encoded_words as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.url_opener(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.localeconv()
    assert module_1.LC_CTYPE == 0
    assert module_1.LC_TIME == 2
    assert module_1.LC_COLLATE == 3
    assert module_1.LC_MONETARY == 4
    assert module_1.LC_MESSAGES == 5
    assert module_1.LC_NUMERIC == 1
    assert module_1.LC_ALL == 6
    assert module_1.CHAR_MAX == 127
    assert module_1.DAY_1 == 131079
    assert module_1.DAY_2 == 131080
    assert module_1.DAY_3 == 131081
    assert module_1.DAY_4 == 131082
    assert module_1.DAY_5 == 131083
    assert module_1.DAY_6 == 131084
    assert module_1.DAY_7 == 131085
    assert module_1.ABDAY_1 == 131072
    assert module_1.ABDAY_2 == 131073
    assert module_1.ABDAY_3 == 131074
    assert module_1.ABDAY_4 == 131075
    assert module_1.ABDAY_5 == 131076
    assert module_1.ABDAY_6 == 131077
    assert module_1.ABDAY_7 == 131078
    assert module_1.MON_1 == 131098
    assert module_1.MON_2 == 131099
    assert module_1.MON_3 == 131100
    assert module_1.MON_4 == 131101
    assert module_1.MON_5 == 131102
    assert module_1.MON_6 == 131103
    assert module_1.MON_7 == 131104
    assert module_1.MON_8 == 131105
    assert module_1.MON_9 == 131106
    assert module_1.MON_10 == 131107
    assert module_1.MON_11 == 131108
    assert module_1.MON_12 == 131109
    assert module_1.ABMON_1 == 131086
    assert module_1.ABMON_2 == 131087
    assert module_1.ABMON_3 == 131088
    assert module_1.ABMON_4 == 131089
    assert module_1.ABMON_5 == 131090
    assert module_1.ABMON_6 == 131091
    assert module_1.ABMON_7 == 131092
    assert module_1.ABMON_8 == 131093
    assert module_1.ABMON_9 == 131094
    assert module_1.ABMON_10 == 131095
    assert module_1.ABMON_11 == 131096
    assert module_1.ABMON_12 == 131097
    assert module_1.RADIXCHAR == 65536
    assert module_1.THOUSEP == 65537
    assert module_1.CRNCYSTR == 262159
    assert module_1.D_T_FMT == 131112
    assert module_1.D_FMT == 131113
    assert module_1.T_FMT == 131114
    assert module_1.AM_STR == 131110
    assert module_1.PM_STR == 131111
    assert module_1.CODESET == 14
    assert module_1.T_FMT_AMPM == 131115
    assert module_1.ERA == 131116
    assert module_1.ERA_D_FMT == 131118
    assert module_1.ERA_D_T_FMT == 131120
    assert module_1.ERA_T_FMT == 131121
    assert module_1.ALT_DIGITS == 131119
    assert module_1.YESEXPR == 327680
    assert module_1.NOEXPR == 327681
    module_0.url_opener(var_0, var_0)

def test_case_2():
    var_0 = 5
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = 'http://example.com'
    var_3 = module_0.url_opener(var_2, var_1)
    assert var_3 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_3():
    var_0 = 'method'
    var_1 = 'encoding'
    var_2 = 'get'
    var_3 = 'utf-8'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    var_6 = module_0._requests(var_5, var_4)
    assert var_6 == '<!doctype html><html lang="en"><head><title>Example Domain</title><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{background:#eee;width:60vw;margin:15vh auto;font-family:system-ui,sans-serif}h1{font-size:1.5em}div{opacity:0.8}a:link,a:visited{color:#348}</style></head><body><div><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission. Avoid use in operations.</p><p><a href="https://iana.org/domains/example">Learn more</a></p></div></body></html>\n'
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_4():
    var_0 = 'method'
    var_1 = 'get'
    var_2 = {var_0: var_1}
    var_3 = 'http://example.com/bad'
    with pytest.raises(module_2.HTTPError):
        module_0._requests(var_3, var_2)

def test_case_5():
    var_0 = 'http://example.com'
    var_1 = 'get'
    var_2 = 'data'
    var_3 = 'k'
    var_4 = 'v'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
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
    var_8 = module_3.urlencode(var_7)
    assert var_8 == 'key=value'
    assert module_3.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_3.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_3.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_3.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_3.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_3.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_3.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_3.MAX_CACHE_SIZE == 20
    var_9 = 'utf-8'
    var_10 = module_4.encode(var_9)
    assert var_10 == '=?utf-8?q?utf-8?='
    assert module_4.ascii_letters == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert module_4.digits == '0123456789'
    var_11 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_7():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_3.urlencode(var_2)
    assert var_3 == 'key=value'
    assert module_3.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_3.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_3.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_3.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_3.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_3.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_3.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_3.MAX_CACHE_SIZE == 20
    var_4 = None
    var_5 = module_0._query(var_4, var_1, var_3)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_8():
    var_0 = 'http://example.com?a=b'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_9():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {var_1: var_2}
    var_6 = module_3.urlencode(var_5)
    assert var_6 == 'key=value'
    assert module_3.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_3.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_3.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_3.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_3.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_3.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_3.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_3.MAX_CACHE_SIZE == 20
    var_7 = module_0._query(var_6, var_4, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_10():
    var_0 = 'http://example.com?existing=1&'
    var_1 = 'GET'
    var_2 = 'data'
    var_3 = 'new'
    var_4 = '2'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0._query(var_0, var_1, var_6)
    assert module_0.HAS_REQUEST is True
    assert module_0.DEFAULT_TIMEOUT == 60
    assert f'{type(module_0.basestring).__module__}.{type(module_0.basestring).__qualname__}' == 'builtins.tuple'
    assert len(module_0.basestring) == 2
    assert module_0.allowed_args == ('auth', 'data', 'headers', 'verify', 'cert', 'config', 'hooks', 'proxies', 'cookies')

def test_case_11():
    var_0 = 'method'
    var_1 = 'timeout'
    var_2 = 'POST'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'http://example.com'
    with pytest.raises(module_2.HTTPError):
        module_0._requests(var_5, var_4)
    assert var_6 == 'success'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'html_content'
    var_1 = lambda : var_0
    module_0._urllib(var_1, var_1)