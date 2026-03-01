# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = -633
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = 3
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = module_0.chunk(var_2, var_3)
    var_3.__contains__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_0, var_0, var_0]
    module_0.Range(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.scanr(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.LazyList(var_0)

def test_case_5():
    var_0 = -5
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = None
    var_5 = module_0.MapList(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.MapList'
    assert var_5.func == []
    assert var_5.list is None

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = var_1.count(var_0)
    assert var_2 == 3
    var_3 = module_0.Range(*var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = b'\xf0\xb4\xc0\xa5b\x1e\xeaN\x9eVe\xc1\xe6'
    var_6 = module_0.scanl(var_2, var_4)
    var_7 = module_0.MapList(var_5, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_7) == 0
    var_8 = var_7.__contains__(var_4)
    assert var_8 is False
    var_9 = module_0.split_by(var_4, criterion=var_4)
    var_10 = list(var_9)
    module_0.scanr(var_0, var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 34
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_2.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = [var_0]
    module_0.Range(*var_1)

def test_case_9():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = var_1.count(var_2)
    assert var_3 == 0
    assert len(var_1) == 0
    var_4 = var_1.count(var_3)
    assert var_4 == 0

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = -5
    var_1 = False
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = None
    var_3.__getitem__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = [var_0]
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

def test_case_12():
    var_0 = -5
    var_1 = True
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = range(var_0)
    var_4 = 3
    var_5 = lambda x: x % var_4 == var_3
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = module_0.LazyList(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_8.iter).__module__}.{type(var_8.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_8.exhausted is False
    assert var_8.list == []
    with pytest.raises(TypeError):
        var_8.__len__()

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1, var_1, var_1]
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.count(var_2)
    assert var_4 == 0
    assert len(var_3) == 4
    var_5 = module_0.scanl(var_0, var_0)
    var_4.__len__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.scanr(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = [var_0]
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
    var_2.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_2)
    assert var_3 is False
    assert len(var_2) == 4
    module_0.scanr(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.scanr(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_2.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3.__contains__(var_0)
    assert var_4 is True
    assert var_3.list == [None]
    var_5 = var_3.count(var_4)
    assert var_5 == 0
    assert len(var_3) == 2
    var_6 = module_0.MapList(var_0, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert var_6.func is None
    assert var_6.list is True
    var_5.__getitem__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_4.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = -5
    var_1 = True
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = range(var_0)
    var_4 = 3
    var_5 = lambda x: x % var_4 == var_3
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = var_7.__contains__(var_4)
    assert var_8 is False
    var_9 = module_0.MapList(var_8, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_9) == 0
    var_9.__getitem__(var_8)

def test_case_22():
    var_0 = 16
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = var_2.__reversed__()
    var_5 = var_2.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5) == 0
    var_6 = module_0.split_by(var_3, var_4, criterion=var_3)
    var_7 = list(var_6)
    var_8 = var_7.__iter__()
    var_9 = module_0.scanr(var_4, var_7)
    var_10 = var_4.__iter__()

def test_case_23():
    var_0 = 3
    var_1 = 4
    var_2 = 5
    var_3 = [var_1, var_2, var_0, var_1, var_2]
    var_4 = module_0.drop(var_0, var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = [var_1, var_0, var_0, var_1, var_2]
    var_7 = module_0.drop(var_2, var_6)
    var_8 = list(var_7)
    var_9 = 0
    var_10 = [var_0, var_8, var_0, var_1, var_2]
    var_11 = module_0.drop(var_9, var_10)
    var_12 = list(var_11)
    var_13 = []
    var_14 = module_0.drop(var_0, var_13)
    var_15 = list(var_14)
    var_16 = 10
    var_17 = [var_0, var_9, var_0]
    var_18 = module_0.drop(var_16, var_17)
    var_19 = list(var_18)
    var_20 = range(var_16)
    var_21 = -1
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.drop(var_21, var_25)
    with pytest.raises(ValueError):
        var_27 = list(var_26)

def test_case_24():
    var_0 = False
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.drop(var_0, var_0)
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_25():
    var_0 = 3
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_26():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 9
    var_6 = range(var_5)
    var_7 = module_0.chunk(var_0, var_6)
    var_8 = list(var_7)
    var_9 = module_0.chunk(var_0, var_4)
    var_10 = list(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = [var_11, var_12, var_0]
    var_14 = module_0.chunk(var_11, var_13)
    var_15 = list(var_14)
    var_16 = 5
    var_17 = range(var_16)
    var_18 = module_0.chunk(var_1, var_17)
    var_19 = list(var_18)
    var_20 = 0
    var_21 = module_0.chunk(var_20, var_9)
    with pytest.raises(ValueError):
        var_22 = list(var_21)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.drop(var_0, var_2)
    var_5 = b'\xf0\xb4\xc0\xa5b\x1e\xeaN\x9eVe\xc1\xe6'
    var_6 = module_0.scanl(var_3, var_4)
    var_7 = module_0.MapList(var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_7) == 0
    var_8 = module_0.split_by(var_4, criterion=var_4)
    var_9 = list(var_8)
    var_7.index(var_6, stop=var_3)

def test_case_28():
    var_0 = 38
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_29():
    var_0 = -5
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_30():
    var_0 = -5
    var_1 = range(var_0)
    var_2 = None
    var_3 = module_0.split_by(var_1, criterion=var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_4 = list(var_3)

def test_case_31():
    var_0 = False
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_32():
    var_0 = -5
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_2 = list(var_1)

def test_case_33():
    var_0 = -5
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = var_2.__iter__()

def test_case_34():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 0
    var_4 = var_3.__contains__(var_3)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = -5
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = False
    var_4 = None
    var_5 = var_2.count(var_4)
    assert var_5 == 0
    var_6 = None
    var_7 = module_0.take(var_3, var_6)
    var_8 = var_2.__iter__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_8) == 0
    var_9 = list(var_8)
    var_5.__len__()

def test_case_36():
    var_0 = -5
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = False
    var_4 = None
    var_5 = var_2.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5) == 0
    var_6 = module_0.take(var_3, var_5)
    var_7 = module_0.split_by(var_6, criterion=var_6)
    var_8 = list(var_7)
    var_9 = var_8.__contains__(var_4)
    assert var_9 is False

def test_case_37():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = module_0.take(var_3, var_4)
    var_6 = module_0.split_by(var_5, criterion=var_5)
    var_7 = list(var_6)
    var_8 = var_4.__contains__(var_4)
    assert var_8 is False

def test_case_38():
    var_0 = 3
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_39():
    var_0 = -11
    var_1 = 1
    var_2 = 4
    var_3 = 9
    var_4 = 16
    var_5 = 25
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 100
    var_8 = range(var_7)
    var_9 = module_0.LazyList(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_9.iter).__module__}.{type(var_9.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_9.exhausted is False
    assert var_9.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_10 = var_9.list
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = var_9[var_0]
    var_13 = var_9.list
    var_14 = len(var_13)
    var_15 = 5
    var_16 = var_9[var_15]
    var_17 = var_9.list
    var_18 = len(var_6)
    var_19 = 20
    var_20 = var_9[var_19]
    var_21 = var_9.list
    var_22 = len(var_21)
    var_23 = range(var_7)
    var_24 = module_0.LazyList(var_23)
    assert len(var_9) == 100
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_24.iter).__module__}.{type(var_24.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_24.exhausted is False
    assert var_24.list == []
    var_25 = var_24.list
    var_26 = len(var_6)
    var_27 = var_24[var_0:var_19]
    var_28 = var_24.list
    var_29 = len(var_28)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = False
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = None
    var_6 = var_2.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_7 = module_0.split_by(var_6, criterion=var_6)
    var_8 = var_2.__getitem__(var_3)
    assert var_8 == 34
    var_3.__getitem__(var_5)

def test_case_41():
    var_0 = 34
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 34
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 34
    var_4 = b'\xf0\xb4\xc0\xa5b\x1e\xeaN\x9eVe\xc1\xe6'
    var_5 = module_0.chunk(var_0, var_2)
    var_6 = module_0.MapList(var_4, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_6) == 34
    var_7 = var_2.__reversed__()
    var_8 = module_0.split_by(var_3, criterion=var_3)
    with pytest.raises(TypeError):
        var_9 = list(var_8)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 34
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = b'\xf0\xb4\xc0b\x1e\xea\x9eVe\xc1\xe6'
    var_6 = module_0.scanl(var_3, var_4)
    var_7 = module_0.MapList(var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_7) == 0
    var_8 = var_2.__reversed__()
    var_9 = module_0.split_by(var_4, criterion=var_4)
    var_10 = list(var_9)
    var_11 = [var_3, var_6]
    module_0.scanr(var_4, var_4, *var_11)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 16
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = var_2.__reversed__()
    var_5 = var_2.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5) == 0
    var_6 = module_0.split_by(var_3, var_4, criterion=var_3)
    var_7 = list(var_6)
    var_8 = b'\xfc\xfbe\x9f\xb3\xad`\xf6'
    module_0.scanr(var_4, var_8, *var_7)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = -618
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = 4412
    var_4 = module_0.drop(var_3, var_2)
    var_5 = var_2.__getitem__(var_0)
    assert var_5 == 381306
    var_6 = var_2.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_7 = module_0.split_by(var_6, criterion=var_6)
    var_8 = list(var_7)
    var_9 = var_2.__iter__()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_9) == 0
    var_10 = var_2.__reversed__()
    var_11 = var_2.__reversed__()
    var_5.__contains__(var_8)

def test_case_45():
    var_0 = 28
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__getitem__(var_0)
    assert var_3 == 812
    var_4 = None
    var_5 = module_0.split_by(var_4, separator=var_3)
    with pytest.raises(TypeError):
        var_6 = list(var_5)