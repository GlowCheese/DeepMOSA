# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0
import genericpath as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, extract=var_0, progress=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, extract=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.download(var_0, extract=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '@Ww1l_{sh\tF\ro\tQ\r7'
    module_0.download(var_0, var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '@Ww1l_{sh\tF\roQ\r7'
    module_0.download(var_0, extract=var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '@Ww1l_{sh\tF\roQ\r7'
    module_0.download(var_0, filename=var_0, extract=var_0, bar_fn=var_0)

def test_case_6():
    var_0 = None
    var_1 = ''
    var_2 = {}
    var_3 = module_0.download(var_1, bar_fn=var_0, **var_2)
    assert var_3 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_7():
    var_0 = 'https://raw.githubusercontent.com/huzecong/flutes/master/README.md'
    var_1 = None
    var_2 = module_0.download(var_0, var_1, progress=var_0)
    assert var_2 == '/tmp/README.md'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_3 = 'All tests passed!'
    var_4 = print(var_3)

def test_case_8():
    var_0 = 'https://drive.google.com/file/d/1-1wAx7b-USG0eQwIBVwVDUl3K1_1ReCt/view?usp=sharing'
    var_1 = 'https://github.com/huzecong/flutes/archive/master.zip'
    var_2 = None
    var_3 = "DMsYRIO#\nhP.XB[Z2'?"
    var_4 = 'wRa'
    var_5 = {var_1: var_2, var_3: var_2, var_4: var_2}
    var_6 = module_0.download(var_0, filename=var_2, progress=var_2, **var_5)
    assert var_6 == '/tmp/1-1wAx7b-USG0eQwIBVwVDUl3K1_1ReCt'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_7 = 'All tests passed!'
    var_8 = print(var_7)

def test_case_9():
    var_0 = 'https://github.com/huzecong/flutes/archive/master.zip'
    var_1 = None
    var_2 = 'yxB> j9/n~=Kq'
    var_3 = 'H\x0c>~qRn!,_U(p'
    var_4 = {var_2: var_1, var_3: var_1}
    var_5 = module_0.download(var_0, bar_fn=var_1, **var_4)
    assert var_5 == '/tmp/master.zip'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_6 = 'All tests passed!'
    var_7 = print(var_6)

def test_case_10():
    var_0 = 'https://raw.githubusercontent.com/huzecong/flutes/master/README.md'
    var_1 = '/tmp'
    var_2 = 'README.md'
    var_3 = module_0.download(var_0, var_1, var_2)
    assert var_3 == '/tmp/README.md'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_4 = module_1.exists(var_3)
    assert var_4 is True
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_5 = 'https://drive.google.com/file/d/1-1wAx7b-USG0eQwIBVwVDUl3K1_1ReCt/view'
    var_6 = '/tmp'
    var_7 = 'test.txt'
    var_8 = module_0.download(var_5, var_6, var_7)
    assert var_8 == '/tmp/test.txt'
    var_9 = module_1.exists(var_8)
    assert var_9 is True
    var_10 = 'https://github.com/huzecong/flutes/archive/master.zip'
    var_11 = '/tmp'
    var_12 = 'flutes-master.zip'
    var_13 = True
    var_14 = module_0.download(var_10, var_11, var_12, var_13)
    assert var_14 == '/tmp/flutes-master.zip'
    var_15 = module_1.exists(var_14)
    assert var_15 is True
    var_16 = 'https://raw.githubusercontent.com/huzecong/flutes/master/README.md'
    var_17 = '/tmp'
    var_18 = 'README.md'
    var_19 = module_0.download(var_16, var_17, var_18, progress=var_13)
    assert var_19 == '/tmp/README.md'
    var_20 = module_1.exists(var_19)
    assert var_20 is True
    var_21 = module_1.exists(var_19)
    assert var_21 is True
    var_22 = 'All tests passed!'
    var_23 = print(var_22)