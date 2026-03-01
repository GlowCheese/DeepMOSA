# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import urllib.parse as module_1
import genericpath as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'Z~&D\x0cFi\x0bc%'
    module_0.download(var_0, filename=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    module_0.download(var_0, var_0, extract=var_0, **var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\t\\l|C\x0bzR96'
    module_0.download(var_0, bar_fn=var_0)

def test_case_4():
    var_0 = '8\tb1lk/'
    var_1 = module_0.download(var_0, bar_fn=var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = module_1.splitvalue(var_1)
    assert module_1.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_1.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_1.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_1.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_1.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_1.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_1.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_1.MAX_CACHE_SIZE == 20

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'aOe^e*Am%Tx'
    module_0.download(var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = "hzSd'5_hZuBE0"
    var_2 = '[Fj'
    var_3 = ''
    var_4 = {var_1: var_0, var_2: var_0, var_3: var_0, var_3: var_0}
    module_0.download(var_2, var_0, extract=var_0, progress=var_4, bar_fn=var_1)

def test_case_7():
    var_0 = None
    var_1 = "enY%'YChh&?H\\C#%C?1"
    var_2 = 'https://drive.google.com/file/d/test_file_id/view'
    var_3 = module_0.download(var_2, var_0, var_1)
    assert var_3 == "/tmp/enY%'YChh&?H\\C#%C?1"
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_2.exists(var_3)
    assert var_4 is True
    assert f'{type(module_2.ALLOW_MISSING).__module__}.{type(module_2.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = 'https://drive.google.com/file/d/test_file_id/view'
    module_0.download(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '9v oYW1Ix2nA TI'
    var_1 = 'https://drive.google.com/f(le/dest_file_id/view'
    var_2 = None
    var_3 = module_0.download(var_1, progress=var_2, bar_fn=var_2)
    assert var_3 == '/tmp/tps:'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.download(var_2, filename=var_0, progress=var_0)