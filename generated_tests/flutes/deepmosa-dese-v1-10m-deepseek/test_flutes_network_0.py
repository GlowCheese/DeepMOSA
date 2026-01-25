# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import re as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.download(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Z~&D\x0cFi\x0bc%'
    module_0.download(var_0, filename=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    module_0.download(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '>('
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'BT_.q`'
    var_1 = None
    module_0.download(var_0, progress=var_0, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\t*K(p_\x0blA_Z~'
    var_1 = '\x0c;{hOb'
    module_0.download(var_0, filename=var_1, progress=var_1, bar_fn=var_1)

def test_case_7():
    var_0 = '/tmp'
    var_1 = module_0.download(var_0, var_0, var_0)
    assert var_1 == '/tmp'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/d/1a2b3c4d5e6f7g8h9i0j/edit?usp=sharing'
    var_1 = None
    var_2 = module_0.download(var_0, var_1)
    assert var_2 == '/tmp/1a2b3c4d5e6f7g8h9i0j'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._extract_google_drive_file_id(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    var_1 = '/tmp'
    module_0.download(var_0, var_1, var_0)

def test_case_10():
    var_0 = 'https://drive.google.com/d///1a2b3c4d5e6f7g8h9i0j///view'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == ''
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_11():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/1A2B3C4D5E6F7G8H9I0J'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view'
    var_1 = '/tmp'
    var_2 = 'mMrn'
    var_3 = module_0.download(var_0, var_1, var_2)
    assert var_3 == '/tmp/mMrn'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._download_from_google_drive(var_2, var_3, var_0, var_3)

def test_case_13():
    var_0 = 'https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view'
    var_1 = '-T_lhR5g'
    var_2 = module_1.RegexFlag.MULTILINE
    var_3 = module_0.download(var_0, filename=var_1, extract=var_2)
    assert var_3 == '/tmp/-T_lhR5g'
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_14():
    var_0 = 'http://example.com'
    var_1 = None
    var_2 = module_0.download(var_0, filename=var_1, progress=var_1)
    assert var_2 == '/tmp/example.com'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'