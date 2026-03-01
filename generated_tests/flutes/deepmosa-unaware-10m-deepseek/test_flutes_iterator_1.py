# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = -6097
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = module_0.LazyList(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = module_0.scanl(var_1, var_4, *var_4)
    assert len(var_4) == 0
    var_7 = list(var_5)
    var_6.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = [var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = -6097
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = var_3.__len__()
    assert var_6 == 0
    var_7 = list(var_5)
    var_2.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = 'C'
    var_2 = module_0.scanr(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.LazyList(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = 'C'
    var_3 = module_0.scanr(var_0, var_2)
    var_3.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = -6097
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = var_2.__iter__()
    var_7 = list(var_5)
    var_6.__getitem__(var_6)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = True
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = [var_2, var_2, var_2, var_2]
    module_0.Range(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = -6097
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = list(var_5)
    var_2.__getitem__(var_2)

def test_case_10():
    var_0 = -6084
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__len__()
    assert var_3 == 0
    var_4 = module_0.LazyList(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = var_4.count(var_1)
    assert var_5 == 0
    assert len(var_4) == 0
    var_6 = var_1.__len__()
    assert var_6 == 0
    var_7 = module_0.split_by(var_1, criterion=var_0)
    var_8 = list(var_7)
    var_9 = var_4.__contains__(var_5)
    assert var_9 is False

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = -197
    var_1 = [var_0, var_0]
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
    var_3.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = -197
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
    var_2.__getitem__(var_2)

def test_case_15():
    var_0 = -1683
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
    var_3 = var_2.count(var_1)
    assert var_3 == 0
    assert len(var_2) == 0

def test_case_16():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(StopIteration):
        var_2.__next__()

def test_case_17():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_3)
    assert var_4 == 0
    assert len(var_3) == 2
    var_5 = var_2.__getitem__(var_4)
    assert var_5 == -197
    var_6 = var_2.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0

def test_case_18():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_3)
    assert var_4 == 0
    assert len(var_3) == 2
    var_5 = var_2.__getitem__(var_4)
    assert var_5 == -197
    var_6 = var_2.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_7 = var_2.__getitem__(var_5)
    assert var_7 == -394

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3.iter) == 0
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.index == 0
    var_5 = var_3.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_5.index == 0
    var_6 = module_0.scanl(var_5, var_5, *var_4)
    assert len(var_3) == 0
    var_2.__getitem__(var_6)

def test_case_20():
    var_0 = -6097
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
    var_3 = var_2.count(var_1)
    assert var_3 == 0
    assert len(var_2) == 0
    var_4 = var_2.__iter__()
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = list(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = -197
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
    var_3 = var_2.count(var_2)
    assert var_3 == 0
    assert len(var_2) == 2
    var_0.__getitem__(var_0)

def test_case_22():
    var_0 = -6097
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
    var_3 = var_2.count(var_1)
    assert var_3 == 0
    assert len(var_2) == 0
    var_4 = module_0.split_by(var_1, criterion=var_0)
    var_5 = var_2.__len__()
    assert var_5 == 0
    var_6 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3.iter) == 0
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_2.__len__()
    assert var_4 == 0
    var_3.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = -197
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
    var_3 = None
    var_2.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 628
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 628
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__reversed__()
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False
    var_2.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.drop_until(var_1, var_1)
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.__contains__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.take(var_0, var_2)
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.count(var_3)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = -6097
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = module_0.split_by(var_1, criterion=var_0)
    var_6 = var_3.__len__()
    assert var_6 == 0
    var_7 = list(var_5)
    var_3.__getitem__(var_4)

def test_case_29():
    var_0 = -2351
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_1, criterion=var_0)
    with pytest.raises(ValueError):
        var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = -197
    var_1 = [var_0, var_0]
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
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4.iter) == 0
    assert var_4.exhausted is False
    assert var_4.list == []
    module_0.scanr(var_3, var_3, *var_1)

def test_case_31():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = module_0.take(var_3, var_2)
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.generator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = None
    var_8 = var_6.__contains__(var_7)
    assert var_8 is False
    assert len(var_6) == 0
    var_9 = var_2.__len__()
    assert var_9 == 0
    var_10 = var_2.__getitem__(var_0)
    assert var_10 == -394

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = False
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = module_0.take(var_3, var_2)
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.generator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = None
    var_8 = var_6.__contains__(var_7)
    assert var_8 is False
    assert len(var_6) == 0
    var_9 = var_2.__len__()
    assert var_9 == 0
    var_9.__iter__()

def test_case_33():
    var_0 = -151
    var_1 = [var_0, var_0]
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
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = module_0.LazyList(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5.iter) == 0
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_4.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_7 = module_0.LazyList(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_7.iter).__module__}.{type(var_7.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_7.iter) == 0
    assert var_7.exhausted is False
    assert var_7.list == []
    var_8 = module_0.LazyList(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_8.iter).__module__}.{type(var_8.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_8.iter) == 0
    assert var_8.exhausted is False
    assert var_8.list == []
    var_9 = var_8.count(var_4)
    assert var_9 == 0
    assert len(var_8) == 0
    var_10 = var_2.__contains__(var_9)
    assert var_10 is False
    var_11 = var_8.__iter__()
    var_12 = 'v'
    var_13 = module_0.scanl(var_10, var_11, *var_12)
    var_14 = module_0.chunk(var_10, var_11)
    var_15 = None
    var_16 = module_0.scanl(var_15, var_15, *var_13)
    var_17 = module_0.scanl(var_16, var_11)
    var_18 = var_4.__getitem__(var_9)
    assert var_18 == -151

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__reversed__()
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False
    var_5 = module_0.LazyList(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.generator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_5.count(var_4)
    assert var_6 == 0
    assert len(var_5) == 0
    var_7 = []
    module_0.scanr(var_6, var_1, *var_7)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    var_4 = module_0.drop(var_3, var_2)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.generator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_5.count(var_4)
    assert var_6 == 0
    assert len(var_5) == 0
    module_0.scanr(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = False
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = module_0.drop(var_3, var_2)
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.generator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = var_6.count(var_5)
    assert var_7 == 0
    assert len(var_6) == 0
    module_0.scanr(var_5, var_5)

def test_case_37():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    var_4 = module_0.chunk(var_3, var_2)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.generator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = None
    var_7 = var_5.__contains__(var_6)
    assert var_7 is False
    assert len(var_5) == 0

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = -134
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.drop(var_0, var_2)
    var_4 = var_2.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = module_0.LazyList(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.generator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_4.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_5.__contains__(var_4)

def test_case_39():
    var_0 = -9
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = module_0.chunk(var_2, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = None
    var_6 = var_4.__contains__(var_5)
    assert var_6 is False
    assert len(var_4) == 2
    var_3.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = True
    var_3 = module_0.take(var_2, var_2)
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = None
    var_4.__contains__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = -197
    var_1 = [var_0, var_0]
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
    var_4 = module_0.chunk(var_3, var_1)
    var_5 = module_0.LazyList(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5.iter) == 0
    assert var_5.exhausted is False
    assert var_5.list == []
    var_5.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = -197
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = True
    var_4 = None
    var_5 = module_0.drop(var_3, var_4)
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.generator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = None
    var_6.__contains__(var_7)

def test_case_44():
    var_0 = 870
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_45():
    var_0 = -2351
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_46():
    var_0 = 657
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_47():
    var_0 = -2351
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_48():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0.scanl(var_0, var_0, *var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.split_by(var_0, criterion=var_0)
    with pytest.raises(ValueError):
        var_4 = list(var_3)

def test_case_49():
    var_0 = 10
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
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = range(var_0)
    var_7 = module_0.LazyList(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_7.iter).__module__}.{type(var_7.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_7.exhausted is False
    assert var_7.list == []
    var_8 = 100
    var_9 = range(var_8)
    var_10 = module_0.LazyList(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_10.iter).__module__}.{type(var_10.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_10.exhausted is False
    assert var_10.list == []
    var_11 = var_10[var_0]
    var_12 = var_10.list
    var_13 = len(var_12)
    assert var_13 == 11
    var_14 = range(var_3)
    var_15 = module_0.LazyList(var_14)
    assert var_10.list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_15.iter).__module__}.{type(var_15.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_15.exhausted is False
    assert var_15.list == []
    var_16 = range(var_3)
    var_17 = module_0.LazyList(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_17.iter).__module__}.{type(var_17.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_17.exhausted is False
    assert var_17.list == []
    var_18 = 10
    with pytest.raises(IndexError):
        var_19 = var_17[var_18]

def test_case_50():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 2
    var_6 = 6
    var_7 = range(var_6)
    var_8 = module_0.chunk(var_5, var_7)
    var_9 = list(var_8)
    var_10 = 5
    var_11 = range(var_0)
    var_12 = module_0.chunk(var_10, var_11)
    var_13 = list(var_12)
    var_14 = 1
    var_15 = range(var_0)
    var_16 = module_0.chunk(var_14, var_15)
    var_17 = list(var_16)
    var_18 = []
    var_19 = module_0.chunk(var_0, var_18)
    var_20 = list(var_19)
    var_21 = 'abcdef'
    var_22 = module_0.chunk(var_5, var_21)
    var_23 = list(var_22)
    var_24 = range(var_10)
    var_25 = 0
    var_26 = 5
    var_27 = range(var_26)
    var_28 = module_0.chunk(var_25, var_27)
    with pytest.raises(ValueError):
        var_29 = list(var_28)

def test_case_51():
    var_0 = 491
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 491
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = None
    var_5 = module_0.split_by(var_1, criterion=var_4, separator=var_1)
    var_6 = list(var_5)

def test_case_52():
    var_0 = 10
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
    var_3 = 5
    var_4 = range(var_3)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = range(var_0)
    var_7 = module_0.LazyList(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_7.iter).__module__}.{type(var_7.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_7.exhausted is False
    assert var_7.list == []
    var_8 = 100
    var_9 = range(var_8)
    var_10 = module_0.LazyList(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_10.iter).__module__}.{type(var_10.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_10.exhausted is False
    assert var_10.list == []
    var_11 = 50
    var_12 = var_10[var_11]
    var_13 = var_10.list
    var_14 = len(var_13)
    assert var_14 == 51
    var_15 = 3
    var_16 = range(var_15)
    var_17 = module_0.LazyList(var_16)
    assert var_10.list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_17.iter).__module__}.{type(var_17.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_17.exhausted is False
    assert var_17.list == []
    var_18 = 0
    var_19 = var_17[var_18]
    var_20 = 1
    var_21 = var_17[var_20]
    var_22 = 2
    var_23 = var_17[var_22]
    var_24 = range(var_3)
    var_25 = module_0.LazyList(var_24)
    assert var_17.list == [0, 1, 2]
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_25.iter).__module__}.{type(var_25.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_25.exhausted is False
    assert var_25.list == []
    var_26 = var_25[var_22:var_0]
    var_27 = []
    var_28 = module_0.LazyList(var_27)
    assert len(var_25) == 5
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_28.iter).__module__}.{type(var_28.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_28.exhausted is False
    assert var_28.list == []
    var_29 = 0
    with pytest.raises(IndexError):
        var_30 = var_28[var_29]