# Check out: https://github.com/GlowCheese/deepmosa
import email._header_value_parser as module_3
import encodings.idna as module_4
import urllib.parse as module_1
import urllib.request as module_2

import isort.core as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = False
    module_0.process(var_0, var_0, raise_on_skip=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_1.unwrap(var_0)
    assert var_1 == 'None'
    assert module_1.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_1.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_1.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_1.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_1.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_1.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_1.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_1.MAX_CACHE_SIZE == 20
    module_0.process(var_1, var_1, raise_on_skip=var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_2.noheaders()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'email.message.Message'
    assert len(var_1) == 1
    assert module_2.MAXFTPCACHE == 10
    assert module_2.ftpcache == {}
    module_0.process(var_1, var_0)
    var_3 = var_2.visit_Assert(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_3.quote_string(var_0)
    assert var_1 == '"None"'
    assert module_3.hexdigits == '0123456789abcdefABCDEF'
    assert module_3.WSP == {' ', '\t'}
    assert module_3.CFWS_LEADER == {' ', '(', '\t'}
    assert module_3.SPECIALS == {'"', '(', ':', '.', '[', '\\', '>', '<', '@', ')', ',', ';', ']'}
    assert module_3.ATOM_ENDS == {'"', '(', '\t', ':', '.', '[', '\\', '>', '<', '@', ')', ',', ';', ' ', ']'}
    assert module_3.DOT_ATOM_ENDS == {'"', '(', '\t', ':', '[', '\\', '>', '<', '@', ')', ',', ';', ' ', ']'}
    assert module_3.PHRASE_ENDS == {':', '[', ']', '\\', '<', '@', ')', ',', ';', '>'}
    assert module_3.TSPECIALS == {'"', '(', ':', '[', '=', '\\', '>', '<', '@', ',', ')', ';', '/', '?', ']'}
    assert module_3.TOKEN_ENDS == {'"', '(', '\t', ':', '[', '=', '\\', ']', '<', '@', ',', ')', ';', '/', '?', ' ', '>'}
    assert module_3.ASPECIALS == {'"', '(', ':', '%', '*', "'", '[', '=', '\\', ']', '<', '@', ',', ')', ';', '/', '?', '>'}
    assert module_3.ATTRIBUTE_ENDS == {'"', '(', ':', '[', ']', "'", '=', '<', '/', ')', '?', '\t', '*', '\\', '@', '>', '%', ',', ';', ' '}
    assert module_3.EXTENDED_ATTRIBUTE_ENDS == {'"', '(', ':', '[', ']', "'", '=', '<', '/', ')', '?', '\t', '*', '\\', '@', '>', ',', ';', ' '}
    assert module_3.NLSET == {'\n', '\r'}
    assert module_3.SPECIALSNL == {'"', '(', ':', '\r', '.', '[', '\\', '>', '<', '@', ')', ',', ';', '\n', ']'}
    assert f'{type(module_3.rfc2047_matcher).__module__}.{type(module_3.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_3.DOT).__module__}.{type(module_3.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.DOT) == 1
    assert f'{type(module_3.ListSeparator).__module__}.{type(module_3.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.ListSeparator) == 1
    assert f'{type(module_3.RouteComponentMarker).__module__}.{type(module_3.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.RouteComponentMarker) == 1
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_4.getregentry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'codecs.CodecInfo'
    assert len(var_0) == 4
    assert f'{type(module_4.unicodedata).__module__}.{type(module_4.unicodedata).__qualname__}' == 'unicodedata.UCD'
    assert f'{type(module_4.dots).__module__}.{type(module_4.dots).__qualname__}' == 're.Pattern'
    assert module_4.ace_prefix == b'xn--'
    assert module_4.sace_prefix == 'xn--'
    var_1 = var_0.__str__()
    var_2 = var_1.__repr__()
    var_3 = 'H1~'
    module_0.process(var_2, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_1.urlparse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'urllib.parse.ParseResultBytes'
    assert len(var_1) == 6
    assert module_1.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_1.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_1.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_1.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_1.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_1.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_1.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_1.MAX_CACHE_SIZE == 20
    module_0.process(var_1, var_1)