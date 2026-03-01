# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = [var_0, var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = -10
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.split_by(var_1, criterion=var_2)
    var_4 = list(var_3)
    var_2.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.scanr(var_0, var_0)

def test_case_4():
    var_0 = 2
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
    with pytest.raises(TypeError):
        var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = -1764
    var_3 = module_0.MapList(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.MapList'
    assert var_3.func == -1764
    assert var_3.list is None
    var_3.__len__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 0
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_2.func).__module__}.{type(var_2.func).__qualname__}' == 'builtins.generator'
    assert f'{type(var_2.list).__module__}.{type(var_2.list).__qualname__}' == 'builtins.generator'
    var_2.__contains__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = [var_0, var_0]
    module_0.scanr(var_0, var_1)

def test_case_8():
    var_0 = 3
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 3
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.count(var_0)
    assert var_3 == 0

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'H}g6fJrx/8$9&BP'
    var_1 = None
    var_2 = [var_0, var_1, var_0, var_0]
    module_0.Range(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '&!!rX\\Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 7
    var_3 = [var_2, var_2]
    var_4 = module_0.Range(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = var_4.__contains__(var_0)
    assert var_5 is False
    var_6 = module_0.MapList(var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert var_6.func is False
    assert var_6.list is False
    var_7 = var_4.__contains__(var_5)
    assert var_7 is False
    var_6.index(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = []
    var_1 = module_0.drop_until(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = None
    var_3.__getitem__(var_4)

def test_case_12():
    var_0 = 2
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
    var_3 = list(var_2)

def test_case_13():
    var_0 = 2604
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
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = -11
    var_6 = var_2[var_5]
    var_7 = range(var_5)
    var_8 = 2921
    var_9 = [x * var_8 for x in var_7]
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.split_by(var_1)
    assert len(var_2) == 2604
    with pytest.raises(ValueError):
        var_13 = list(var_12)

def test_case_14():
    var_0 = -1666
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
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    assert len(var_3) == 0
    var_5 = list(var_2)
    var_6 = var_3.__contains__(var_1)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = -1666
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
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    assert len(var_3) == 0
    var_5 = list(var_2)
    var_3.__getitem__(var_4)

def test_case_16():
    var_0 = -1666
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
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    assert len(var_3) == 0
    var_5 = list(var_2)
    var_6 = var_3.__len__()
    assert var_6 == 0

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '!!NK`rX\\.Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_2.index == 0
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = module_0.split_by(var_3, criterion=var_1)
    var_3.__len__()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = {}
    var_1 = None
    var_2 = module_0.take(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.chunk(var_0, var_1)
    var_4 = [var_1, var_1, var_1]
    module_0.scanr(var_1, var_0, *var_4)

def test_case_19():
    var_0 = 3
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

def test_case_20():
    var_0 = '&!!N`rX\\Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 9
    var_3 = [var_2, var_2]
    var_4 = module_0.Range(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = var_4.__contains__(var_0)
    assert var_5 is False
    var_6 = var_4.__getitem__(var_5)
    assert var_6 == 0
    var_7 = var_4.__contains__(var_2)
    assert var_7 is False
    var_8 = var_4.__len__()
    assert var_8 == 0

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = '&!!N`rX\\Z'
    var_1 = var_0.count(var_0)
    assert var_1 == 1
    var_2 = [var_1, var_1]
    var_3 = module_0.Range(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3.__getitem__(var_0)

def test_case_22():
    var_0 = '&!!N`rX\\Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 9
    var_3 = [var_2, var_2]
    var_4 = module_0.Range(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 0
    var_5 = var_4.__getitem__(var_2)
    assert var_5 == 0

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = b'\xfd'
    var_1 = module_0.scanr(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    module_0.LazyList(var_2)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = '!!NK`rXd.Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 10
    var_3 = [var_2, var_2]
    var_4 = module_0.drop_until(var_2, var_2)
    var_5 = module_0.Range(*var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5) == 0
    var_6 = -2137
    var_7 = var_5.__getitem__(var_6)
    assert var_7 == -2137
    var_2.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = '!!NK`rX\\.Z'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 10
    var_3 = b'\xcf\xaa\xec<$\xebcR\xf3{!\xd7'
    var_4 = None
    var_5 = [var_2, var_2]
    var_6 = module_0.Range(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 0
    var_7 = var_6.__contains__(var_3)
    assert var_7 is False
    var_8 = var_6.__iter__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_8) == 0
    var_9 = False
    var_10 = {var_7: var_8, var_9: var_6, var_2: var_4}
    var_11 = [var_4]
    module_0.scanr(var_8, var_10, *var_11)

def test_case_26():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_27():
    var_0 = 24
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_28():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_29():
    var_0 = -8
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_30():
    var_0 = 0
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_31():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_1.__len__()
    assert var_3 == 5
    var_4 = module_0.take(var_0, var_2)
    with pytest.raises(ValueError):
        var_5 = list(var_4)

def test_case_32():
    var_0 = 0
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_1.__len__()
    assert var_3 == 0
    var_4 = module_0.take(var_0, var_2)
    var_5 = list(var_4)

def test_case_33():
    var_0 = -252
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_34():
    var_0 = -4
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_1, criterion=var_0)
    with pytest.raises(ValueError):
        var_3 = list(var_1)

def test_case_35():
    var_0 = -1908
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.generator'
    assert var_2.exhausted is False
    assert var_2.list == []
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_36():
    var_0 = 1
    var_1 = None
    var_2 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_37():
    var_0 = -1908
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.generator'
    assert var_2.exhausted is False
    assert var_2.list == []
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_38():
    var_0 = 1
    var_1 = None
    var_2 = [var_1, var_1, var_1, var_1]
    var_3 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.drop(var_0, var_2)
    var_5 = module_0.take(var_1, var_3)
    var_6 = list(var_4)

def test_case_39():
    var_0 = 1
    var_1 = None
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_40():
    var_0 = 1
    var_1 = None
    var_2 = module_0.drop(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = [var_3, var_3, var_3, var_3]
    var_5 = module_0.take(var_0, var_3)
    var_6 = module_0.chunk(var_0, var_4)
    var_7 = list(var_6)

def test_case_41():
    var_0 = 2604
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
    var_3 = 10
    var_4 = var_2[var_3]
    var_5 = var_2[var_3]
    var_6 = range(var_0)
    var_7 = 5
    var_8 = range(var_7)
    var_9 = module_0.split_by(var_1)
    assert var_2.list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        var_10 = list(var_9)

def test_case_42():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = 0
    var_9 = [var_1, var_2, var_0]
    var_10 = module_0.take(var_8, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_0]
    var_13 = module_0.take(var_4, var_12)
    var_14 = list(var_13)
    var_15 = 10
    var_16 = range(var_15)
    var_17 = module_0.take(var_4, var_16)
    var_18 = list(var_17)
    var_19 = range(var_15)
    var_20 = module_0.take(var_8, var_19)
    var_21 = list(var_20)
    var_22 = 100
    var_23 = [var_1, var_2, var_0]
    var_24 = module_0.take(var_22, var_23)
    var_25 = list(var_24)
    var_26 = []
    var_27 = module_0.take(var_4, var_26)
    var_28 = -1
    var_29 = 1
    var_30 = 3
    var_31 = [var_29, var_2, var_30]
    var_32 = module_0.take(var_28, var_31)
    with pytest.raises(ValueError):
        var_33 = list(var_32)

def test_case_43():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.drop(var_0, var_6)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_8 = list(var_7)
    var_9 = [var_1, var_2, var_3, var_4, var_5]
    var_10 = module_0.drop(var_5, var_9)
    var_11 = list(var_10)
    var_12 = [var_1, var_2, var_3, var_4, var_5]
    var_13 = module_0.drop(var_2, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.drop(var_3, var_15)
    var_17 = list(var_16)
    var_18 = 10
    var_19 = [var_1, var_2, var_3]
    var_20 = module_0.drop(var_18, var_19)
    var_21 = list(var_20)
    var_22 = -1
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.drop(var_22, var_26)
    with pytest.raises(ValueError):
        var_28 = list(var_27)

def test_case_44():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = 5
    var_5 = range(var_1)
    var_6 = module_0.chunk(var_4, var_5)
    var_7 = list(var_6)
    var_8 = []
    var_9 = module_0.chunk(var_0, var_8)
    var_10 = list(var_9)
    var_11 = range(var_4)
    var_12 = var_2.__iter__()
    var_13 = list(var_12)
    var_14 = 1
    var_15 = range(var_4)
    var_16 = module_0.chunk(var_14, var_15)
    var_17 = list(var_16)
    var_18 = 2
    var_19 = 'b'
    var_20 = 'c'
    var_21 = 'd'
    var_22 = 'e'
    var_23 = [var_7, var_19, var_20, var_21, var_22]
    var_24 = module_0.chunk(var_18, var_23)
    var_25 = list(var_24)
    var_26 = 0
    var_27 = 10
    var_28 = range(var_27)
    var_29 = module_0.chunk(var_26, var_28)
    with pytest.raises(ValueError):
        var_30 = list(var_29)

def test_case_45():
    var_0 = -13
    var_1 = range(var_0)
    var_2 = None
    var_3 = module_0.split_by(var_1, criterion=var_2, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)

def test_case_46():
    var_0 = 3
    var_1 = 0
    var_2 = range(var_1)
    var_3 = True
    var_4 = lambda x: x % var_0 == var_1
    var_5 = module_0.split_by(var_2, var_3, criterion=var_4)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = list(var_5)
    var_7 = ' Split by: '
    var_8 = module_0.split_by(var_7, separator=var_6)
    var_9 = list(var_8)
    var_10 = module_0.split_by(var_7, var_3, separator=var_6)
    var_11 = list(var_10)
    var_12 = []
    var_13 = lambda x: x % var_0 == var_1
    var_14 = module_0.split_by(var_12, criterion=var_13)
    var_15 = list(var_14)
    var_16 = []
    var_17 = module_0.split_by(var_16, separator=var_7)
    var_18 = list(var_17)
    var_19 = 6
    var_20 = 9
    var_21 = [var_0, var_19, var_20]
    var_22 = lambda x: x % var_0 == var_1
    var_23 = module_0.split_by(var_21, criterion=var_22)
    with pytest.raises(NameError):
        var_24 = list(var_23)

def test_case_47():
    var_0 = 3
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = range(var_1)
    var_4 = True
    var_5 = lambda x: x % var_0 == var_1
    var_6 = module_0.split_by(var_3, var_4, criterion=var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = ' Split by: '
    var_9 = ' '
    var_10 = module_0.split_by(var_8, separator=var_9)
    var_11 = list(var_10)
    var_12 = module_0.split_by(var_8, var_4, separator=var_9)
    var_13 = list(var_12)
    var_14 = []
    var_15 = lambda x: x % var_0 == var_1
    var_16 = module_0.split_by(var_14, criterion=var_15)
    var_17 = list(var_16)
    var_18 = []
    var_19 = module_0.split_by(var_18, separator=var_9)
    var_20 = list(var_19)
    var_21 = 6
    var_22 = 9
    var_23 = [var_0, var_21, var_22]
    var_24 = lambda x: x % var_0 == var_1
    var_25 = module_0.split_by(var_23, criterion=var_24)
    with pytest.raises(NameError):
        var_26 = list(var_25)