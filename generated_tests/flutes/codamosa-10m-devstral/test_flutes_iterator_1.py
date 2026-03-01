# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = None
    var_2 = module_0.MapList(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert var_2.func is None
    assert var_2.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.drop_until(var_3, var_0)
    var_5 = [var_0, var_1, var_0]
    module_0.Range(*var_5)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = [var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_2.index(var_0, stop=var_0)

@pytest.mark.xfail(strict=True)
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
    var_1.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.scanr(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.LazyList(var_0)

def test_case_8():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    assert len(var_1) == 13
    var_4 = module_0.MapList(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.MapList'
    assert var_4.func is False
    assert var_4.list is None

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 0
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_0 == var_0
    var_3 = module_0.split_by(var_1, criterion=var_2)
    var_4 = module_0.MapList(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_4.func).__module__}.{type(var_4.func).__qualname__}' == 'builtins.generator'
    assert f'{type(var_4.list).__module__}.{type(var_4.list).__qualname__}' == 'builtins.generator'
    var_5 = None
    var_6 = module_0.take(var_0, var_2)
    var_7 = module_0.MapList(var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert var_7.func is None
    module_0.scanr(var_2, var_7)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 0
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_0 == var_0
    var_3 = module_0.MapList(var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_3.func).__module__}.{type(var_3.func).__qualname__}' == 'builtins.generator'
    assert f'{type(var_3.list).__module__}.{type(var_3.list).__qualname__}' == 'builtins.generator'
    var_3.__contains__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = module_0.MapList(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.MapList'
    assert var_3.func is True
    assert var_3.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = None
    var_5 = module_0.drop_until(var_0, var_0)
    var_6 = [var_1, var_4]
    module_0.Range(*var_6)

def test_case_12():
    var_0 = -22
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = 3252
    var_4 = module_0.LazyList(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.split_by(var_1, var_3, criterion=var_0)
    var_6 = var_4.__contains__(var_1)
    assert var_6 is False
    assert len(var_4) == 0
    var_7 = list(var_5)
    var_8 = module_0.LazyList(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_8.iter).__module__}.{type(var_8.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_8.exhausted is False
    assert var_8.list == []

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_2.index == 0
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    assert len(var_1) == 13
    module_0.scanr(var_3, var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_1.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 0
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_1, criterion=var_1)
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__contains__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_2.index == 0
    var_3 = module_0.drop_until(var_2, var_1)
    var_4 = module_0.scanl(var_2, var_0, *var_2)
    assert len(var_1) == 13
    assert var_2.index == 13
    var_5 = None
    var_6 = module_0.MapList(var_5, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_6) == 13
    var_7 = var_1.__contains__(var_2)
    assert var_7 is False
    module_0.scanr(var_7, var_1)

def test_case_17():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_2.index == 0
    var_3 = var_2.__next__()
    assert var_3 == 190
    assert var_1.list == [190]
    assert var_2.index == 1
    var_4 = None
    var_5 = module_0.MapList(var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.MapList'
    assert var_5.func is None
    assert f'{type(var_5.list).__module__}.{type(var_5.list).__qualname__}' == 'flutes.iterator.LazyList'
    var_6 = var_1.__contains__(var_2)
    assert var_6 is False
    assert len(var_1) == 13
    assert len(var_5) == 13
    module_0.scanr(var_6, var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = b'\x0b'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.drop_until(var_0, var_0)
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    assert len(var_1) == 1
    var_4 = module_0.scanr(var_3, var_1)
    var_5 = var_4.__len__()
    assert var_5 == 1
    var_3.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = b'\xbe\x9aV\xcdg\xe5\xd1u#\xad\x96\xf7F'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reversed__()
    var_3 = var_2.__iter__()
    var_4 = var_1.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.index == 0
    var_5 = var_4.__next__()
    assert var_5 == 190
    assert var_1.list == [190]
    assert var_4.index == 1
    var_1.__getitem__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = -29
    var_1 = range(var_0)
    var_2 = -16
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = list(var_5)
    var_7 = [var_1, var_1, var_1, var_6]
    module_0.scanr(var_3, var_6, *var_7)

def test_case_22():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = list(var_3)
    var_5 = module_0.split_by(var_4, separator=var_4)
    var_6 = list(var_5)
    var_7 = 10
    var_8 = range(var_7)
    var_9 = 'hello'
    var_10 = module_0.take(var_8, var_9)

def test_case_23():
    var_0 = 24
    var_1 = 4
    var_2 = 5
    var_3 = [var_2, var_0, var_0, var_1, var_2]
    var_4 = module_0.take(var_0, var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = list(var_5)
    var_7 = module_0.split_by(var_6, separator=var_6)
    var_8 = list(var_7)
    var_9 = 10
    var_10 = range(var_9)
    var_11 = 1
    var_12 = module_0.take(var_0, var_5)
    var_13 = module_0.drop_until(var_8, var_5)
    var_14 = []
    var_15 = module_0.take(var_6, var_14)
    with pytest.raises(TypeError):
        var_16 = list(var_11)

def test_case_24():
    var_0 = -12
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = -17
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    with pytest.raises(ValueError):
        var_6 = list(var_5)

def test_case_25():
    var_0 = 3255
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_26():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_27():
    var_0 = -22
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_28():
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
    var_12 = var_7.count(var_11)
    assert var_12 == 0
    var_13 = module_0.take(var_4, var_12)
    with pytest.raises(TypeError):
        var_14 = list(var_13)

def test_case_29():
    var_0 = 27
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = -17
    var_3 = -11
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    with pytest.raises(TypeError):
        var_6 = list(var_5)

def test_case_30():
    var_0 = -7
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_1 == var_0
    var_3 = None
    var_4 = module_0.split_by(var_3, criterion=var_3)
    with pytest.raises(ValueError):
        var_5 = list(var_4)

def test_case_31():
    var_0 = -22
    var_1 = range(var_0)
    var_2 = var_1.__iter__()
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.drop_until(var_1, var_2)
    var_5 = var_3.__contains__(var_1)
    assert var_5 is False
    assert len(var_3) == 0
    var_6 = list(var_4)

def test_case_32():
    var_0 = -29
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.scanl(var_1, var_1)
    var_3 = module_0.drop_until(var_1, var_1)
    with pytest.raises(ValueError):
        var_4 = list(var_3)

def test_case_33():
    var_0 = -21
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_1 == var_1
    var_3 = None
    var_4 = module_0.split_by(var_3, criterion=var_3, separator=var_2)
    with pytest.raises(TypeError):
        var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 0
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__contains__(var_1)

def test_case_35():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 20
    var_6 = range(var_1)
    var_7 = module_0.chunk(var_5, var_6)
    var_8 = list(var_7)
    var_9 = 1
    var_10 = 5
    var_11 = range(var_10)
    var_12 = module_0.chunk(var_9, var_11)
    var_13 = list(var_12)
    var_14 = []
    var_15 = module_0.chunk(var_0, var_14)
    var_16 = list(var_15)
    var_17 = range(var_10)
    var_18 = module_0.chunk(var_10, var_17)
    var_19 = list(var_18)
    var_20 = 2
    var_21 = 'a'
    var_22 = 'b'
    var_23 = 'c'
    var_24 = 'd'
    var_25 = 'e'
    var_26 = [var_21, var_22, var_23, var_24, var_25]
    var_27 = module_0.chunk(var_20, var_26)
    var_28 = list(var_27)
    var_29 = range(var_1)
    var_30 = 0
    var_31 = 10
    var_32 = range(var_31)
    var_33 = module_0.chunk(var_30, var_32)
    with pytest.raises(ValueError):
        var_34 = list(var_33)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = [var_0]
    module_0.scanr(var_0, var_1, *var_1)

def test_case_37():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_0]
    var_9 = module_0.drop(var_0, var_8)
    var_10 = list(var_9)
    var_11 = [var_1, var_2, var_0]
    var_12 = module_0.drop(var_4, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = range(var_4)
    var_16 = module_0.drop(var_14, var_15)
    var_17 = list(var_16)
    var_18 = -1
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.drop(var_18, var_22)
    with pytest.raises(ValueError):
        var_24 = list(var_23)

def test_case_38():
    var_0 = -22
    var_1 = range(var_0)
    var_2 = module_0.MapList(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = 3252
    var_4 = module_0.LazyList(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.split_by(var_1, var_3, criterion=var_0)
    var_6 = var_4.__contains__(var_1)
    assert var_6 is False
    assert len(var_4) == 0
    var_7 = list(var_5)