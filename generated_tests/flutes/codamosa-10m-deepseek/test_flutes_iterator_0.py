# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    module_0.Range(*var_1)

def test_case_1():
    var_0 = -892
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert var_2.l == 0
    assert var_2.r == -892
    assert var_2.step == 1
    assert var_2.val == 0
    assert var_2.length == -892
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__len__()
    assert var_3 == -892
    var_4 = module_0.split_by(var_2, criterion=var_2)
    var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.scanr(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.LazyList(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert var_2.func is None
    assert var_2.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.count(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_1.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.index == 0
    var_4 = var_3.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.index == 0
    var_5 = module_0.MapList(var_0, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_5) == 3
    var_6 = var_5.__len__()
    assert var_6 == 3
    var_7 = var_2.__contains__(var_4)
    assert var_7 is False
    assert len(var_2) == 3
    var_8 = module_0.scanl(var_0, var_0, *var_1)
    var_5.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'zDo5-%`H;fTW+'
    var_1 = b''
    module_0.scanr(var_0, var_1)

def test_case_10():
    var_0 = -892
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.index(var_0, var_0)
    assert var_3 == 0

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.index(var_1)

def test_case_13():
    var_0 = -892
    var_1 = [var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_0)
    assert var_3 is True
    assert var_2.list == [-892]

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = var_1.__reversed__()
    var_5 = [var_1, var_1, var_1, var_1]
    module_0.scanr(var_1, var_3, *var_5)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    assert len(var_3) == 0
    module_0.scanr(var_1, var_4)

def test_case_17():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.split_by(var_1, criterion=var_1)
    var_5 = list(var_2)
    var_6 = var_1.__contains__(var_3)
    assert var_6 is False
    assert len(var_2) == 0

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = -878
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.generator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_2.index(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    module_0.scanr(var_0, var_1)

def test_case_20():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.scanl(var_4, var_3)
    var_6 = var_4.__contains__(var_1)
    assert var_6 is False
    assert len(var_4) == 0
    var_7 = var_4.__contains__(var_1)
    assert var_7 is False

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = -878
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4.iter) == 0
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.chunk(var_3, var_3)
    var_6 = module_0.LazyList(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = var_2.__len__()
    assert var_7 == 0
    var_8 = range(var_0)
    var_9 = module_0.split_by(var_8, criterion=var_8)
    var_10 = module_0.split_by(var_3, var_3, criterion=var_6, separator=var_1)
    var_11 = module_0.drop_until(var_8, var_1)
    var_12 = list(var_4)
    var_13 = module_0.scanl(var_2, var_8, *var_8)
    assert len(var_4) == 0
    module_0.scanr(var_8, var_12, *var_6)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.index == 0
    var_4 = var_2.count(var_3)
    assert var_4 == 0
    assert len(var_2) == 3
    var_5 = var_2.__len__()
    assert var_5 == 3
    var_6 = module_0.take(var_4, var_2)
    module_0.LazyList(var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_0)
    assert var_3 is True
    assert var_2.list == [None]
    var_4 = var_2.count(var_2)
    assert var_4 == 0
    assert len(var_2) == 3
    var_4.__getitem__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.index == 0
    var_4 = var_3.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.index == 0
    var_5 = module_0.MapList(var_0, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_5) == 3
    var_6 = var_2.__contains__(var_2)
    assert var_6 is False
    assert len(var_2) == 3
    var_7 = module_0.scanl(var_0, var_0, *var_1)
    var_5.__getitem__(var_0)

def test_case_25():
    var_0 = -878
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_1.__iter__()
    var_4 = var_1.__iter__()
    var_5 = list(var_2)
    var_6 = [var_1]
    var_7 = module_0.scanr(var_1, var_5, *var_6)
    var_8 = var_7.__contains__(var_5)
    assert var_8 is False

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 1
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_0, var_1, var_2, var_1]
    var_4 = module_0.drop(var_0, var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = module_0.drop(var_2, var_3)
    var_7 = list(var_6)
    var_8 = module_0.chunk(var_0, var_5)
    var_9 = list(var_8)
    var_9.__getitem__(var_7)

def test_case_27():
    var_0 = 3
    var_1 = 4
    var_2 = [var_1, var_1, var_0, var_0, var_1]
    var_3 = module_0.drop(var_1, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = module_0.scanr(var_4, var_4)
    var_6 = list(var_5)
    var_7 = [var_6, var_0, var_0, var_1, var_4]
    var_8 = module_0.drop(var_0, var_7)
    var_9 = list(var_8)
    var_10 = 10
    var_11 = [var_6, var_1, var_0, var_1, var_1]
    var_12 = module_0.drop(var_10, var_11)
    var_13 = list(var_12)
    var_14 = []
    var_15 = module_0.drop(var_1, var_14)
    var_16 = list(var_15)

def test_case_28():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 5
    var_4 = []
    var_5 = module_0.drop(var_3, var_4)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = list(var_5)
    var_7 = 4
    var_8 = [var_0, var_1, var_2, var_7]
    var_9 = module_0.drop(var_1, var_8)
    var_10 = list(var_9)
    var_11 = list(var_5)
    var_12 = range(var_3)
    var_13 = -1
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.drop(var_13, var_17)
    with pytest.raises(ValueError):
        var_19 = list(var_18)

def test_case_29():
    var_0 = 10
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    with pytest.raises(TypeError):
        var_6 = list(var_5)

def test_case_30():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * var_1
    var_7 = module_0.MapList(var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_7) == 5
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_8 = 0
    with pytest.raises(NameError):
        var_9 = var_7[var_8]

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_1, var_0, var_1, var_2, var_1]
    var_4 = module_0.drop(var_0, var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = 0
    var_7 = list(var_4)
    var_8 = [var_2, var_0, var_1, var_2, var_6]
    var_9 = module_0.drop(var_2, var_8)
    var_10 = list(var_9)
    var_11 = None
    var_12 = module_0.MapList(var_7, var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_12) == 0
    var_13 = module_0.chunk(var_11, var_12)
    var_14 = module_0.chunk(var_2, var_5)
    var_15 = list(var_14)
    var_15.__getitem__(var_10)

def test_case_32():
    var_0 = 6
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = -898
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.scanr(var_1, var_2, *var_2)

def test_case_34():
    var_0 = 16
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 0
    var_6 = range(var_1)
    var_7 = module_0.take(var_5, var_6)
    var_8 = list(var_7)
    var_9 = module_0.take(var_1, var_7)
    var_10 = list(var_9)

def test_case_35():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = list(var_2)
    var_4 = 0
    var_5 = range(var_1)
    var_6 = module_0.take(var_4, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = range(var_0)
    var_9 = module_0.take(var_1, var_8)

def test_case_36():
    var_0 = 26
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_1 == var_1
    var_3 = module_0.split_by(var_1, criterion=var_2)
    with pytest.raises(TypeError):
        var_4 = list(var_3)

def test_case_37():
    var_0 = 0
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_38():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_39():
    var_0 = -17
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_2 = list(var_1)

def test_case_40():
    var_0 = -17
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_2 = list(var_1)

def test_case_41():
    var_0 = None
    var_1 = None
    var_2 = module_0.split_by(var_0, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_42():
    var_0 = -884
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = -878
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.generator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = None
    var_4 = module_0.scanl(var_2, var_3)
    var_2.__contains__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = -4868
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_1)
    assert var_3 is False
    module_0.scanl(var_3, var_1, *var_3)

def test_case_45():
    var_0 = -892
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert var_2.l == 0
    assert var_2.r == -892
    assert var_2.step == 1
    assert var_2.val == 0
    assert var_2.length == -892
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__len__()
    assert var_3 == -892
    var_4 = var_2.__getitem__(var_3)
    assert var_4 == -1784
    with pytest.raises(TypeError):
        var_5 = list(var_3)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = -892
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__len__()
    assert var_3 == 0
    var_4 = module_0.split_by(var_2, criterion=var_2)
    var_5 = var_4.__iter__()
    var_6 = module_0.LazyList(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = var_2.__getitem__(var_3)
    assert var_7 == -892
    var_6.__getitem__(var_7)

def test_case_47():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_6.exhausted is False
    assert var_6.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = [var_3, var_1, var_2, var_3, var_4]
    var_8 = iter(var_7)
    var_9 = module_0.LazyList(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_9.iter).__module__}.{type(var_9.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_9.exhausted is False
    assert var_9.list == []
    var_10 = 5
    with pytest.raises(IndexError):
        var_11 = var_9[var_10]