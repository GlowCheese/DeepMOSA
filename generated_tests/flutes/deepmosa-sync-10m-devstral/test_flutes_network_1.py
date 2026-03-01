# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'https//drive.google.com/file/dinvalidid/iew'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/tps'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.download(var_1, progress=var_1, bar_fn=var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ' 5r4n^]%L O9LR_&.1'
    module_0.download(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'https/drive.google.com/file/d/iXv!l=did/iew'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/iXv!l=did'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.download(var_0, var_1, var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = {}
    module_0.download(var_0, filename=var_1, **var_1)

def test_case_5():
    pass

def test_case_6():
    var_0 = ''
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'W":z}F Pc!K5)H'
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1}
    module_0.download(var_0, var_1, progress=var_0, **var_2)

def test_case_8():
    var_0 = 'https/drive.google.com/file/d/iXv!lidid/iew'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/iXv!lidid'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_9():
    var_0 = 'https://eample.com/file.txt'
    var_1 = '/tmp'
    var_2 = 'P}MQCYQ\n'
    var_3 = True
    var_4 = {}
    var_5 = module_0.download(var_0, var_1, var_2, progress=var_3, **var_4)
    assert var_5 == '/tmp/P}MQCYQ\n'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_6 = module_1.exists(var_5)
    assert var_6 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_7 = bool(var_6)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'https://drive.google.com/file/d/invalid_id/view'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/invalid_id'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0._download_from_google_drive(var_1, var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'htps://drive.google.com/file/d/inalid_id/view'
    var_1 = module_0.download(var_0, progress=var_0)
    assert var_1 == '/tmp/inalid_id'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_1.readinto(var_1)

def test_case_12():
    var_0 = 'https://drive.gootle.com/fBle/d/invalid_id/virw'
    var_1 = module_0.download(var_0)
    assert var_1 == '/tmp/virw'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'