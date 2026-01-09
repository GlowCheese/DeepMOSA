# Check out: https://github.com/GlowCheese/deepmosa
import flutes.network as module_0
import pytest


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
    var_0 = 'https://drive.google.com/file/d/1abc123def456/view'
    var_1 = '/tmp/test'
    var_2 = False
    var_3 = {var_1: var_1}
    module_0.download(var_0, var_1, var_0, progress=var_2, **var_3)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'Pg!'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'BT_.q`'
    var_1 = None
    module_0.download(var_0, progress=var_0, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '\t*K(p_\x0blA_Z~'
    var_1 = '\x0c;{hOb'
    module_0.download(var_0, filename=var_1, progress=var_1, bar_fn=var_1)

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/abc123/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/abc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://drive.google.com/file/d/aAc123/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/aAc123'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'https://drive.google.com/file/d/1aSc1l3def4h6/view'
    module_0.download(var_0, filename=var_0, extract=var_0)

def test_case_9():
    var_0 = 'https://drive.google.com/file/d/1aSc1l3def4h6/view'
    var_1 = 'BP ?,)T%I9'
    var_2 = module_0.download(var_0, filename=var_1, extract=var_0)
    assert var_2 == '/tmp/BP ?,)T%I9'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_10():
    var_0 = 'https://drive.google.com/file/d/1abc123def456/view'
    var_1 = '/tmp/test'
    var_2 = '$'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    assert var_5 == '/tmp/test/$'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'https://drive.google.com/file/d/1abc123def456/view'
    var_1 = '/tmp/test'
    var_2 = True
    var_3 = {}
    module_0.download(var_0, var_1, var_0, progress=var_2, **var_3)