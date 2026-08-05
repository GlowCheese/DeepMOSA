# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.Range()

def test_case_1():
    var_0 = -25
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__getitem__(var_0)
    assert var_3 == 600

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.scanr(var_0, var_0)

def test_case_3():
    var_0 = 'Q'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    assert len(var_1) == 1

def test_case_4():
    var_0 = 2343
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.MapList(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.MapList'
    assert var_3.func == 2343
    assert var_3.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.split_by(var_1, criterion=var_0)
    with pytest.raises(TypeError):
        var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.Range(*var_1)

def test_case_6():
    var_0 = 2287
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 2287
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_4 = None
    var_5 = module_0.split_by(var_3, criterion=var_4, separator=var_0)
    var_6 = list(var_5)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 966
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = module_0.MapList(var_3, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert var_6.func is None
    assert f'{type(var_6.list).__module__}.{type(var_6.list).__qualname__}' == 'flutes.iterator.LazyList'
    var_7 = var_4.__iter__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_7.index == 0
    var_8 = var_6.__iter__()
    var_4.__contains__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -25
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
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
def test_case_10():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_1.__len__()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = module_0.scanl(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    var_4 = module_0.MapList(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_4.func).__module__}.{type(var_4.func).__qualname__}' == 'builtins.generator'
    assert f'{type(var_4.list).__module__}.{type(var_4.list).__qualname__}' == 'builtins.generator'
    var_5 = 1809
    var_6 = module_0.drop(var_5, var_2)
    var_4.index(var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = "DS;$W#zqNIhT1O'mE"
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    assert len(var_2) == 17
    var_4 = var_2.__iter__()
    var_5 = module_0.drop_until(var_3, var_0)
    module_0.Range(*var_3)

def test_case_13():
    var_0 = 'g!EEk'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_1.__len__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 966
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 1008
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.__getitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = '`'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    assert len(var_2) == 1
    var_4 = var_2.__reversed__()
    var_5 = module_0.scanr(var_4, var_1)
    var_6 = var_2.__len__()
    assert var_6 == 1
    var_6.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = 'VZ"|~\tXM'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.index == 0
    var_4 = var_2.__contains__(var_0)
    assert var_4 is False
    assert len(var_2) == 8
    module_0.Range(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 966
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.count(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 960
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 984
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_4 = module_0.LazyList(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_5 = var_4.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_5.index == 0
    module_0.scanl(var_5, var_3, *var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = "DS;$W#qNIhT1O'mE"
    module_0.scanr(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = '&>sI$}1\\)M{_zUvv=\x0b'
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
    var_3 = None
    module_0.scanr(var_3, var_1, *var_2)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = "DS;HW#qNHkIhT1O'mE"
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
    var_3 = var_2.__next__()
    assert var_3 == 'D'
    assert var_1.list == ['D']
    assert var_2.index == 1
    var_4 = var_1.__contains__(var_2)
    assert var_4 is False
    assert len(var_1) == 18
    var_2.__len__()

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 14
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = module_0.scanr(var_3, var_3)
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = '`'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    assert len(var_3) == 1
    var_5 = var_3.__iter__()
    var_6 = var_3.__reversed__()
    var_7 = module_0.scanr(var_6, var_2)
    module_0.scanr(var_4, var_7, *var_6)

def test_case_26():
    var_0 = 2
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = 40
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = 0
    var_9 = 3
    var_10 = [var_8, var_0, var_9]
    var_11 = module_0.drop(var_8, var_10)
    var_12 = list(var_11)
    var_13 = list(var_11)
    var_14 = module_0.drop(var_9, var_13)
    var_15 = list(var_14)
    var_16 = []
    var_17 = module_0.drop(var_0, var_16)
    var_18 = list(var_17)
    var_19 = -1
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.drop(var_19, var_23)
    with pytest.raises(ValueError):
        var_25 = list(var_24)

def test_case_27():
    var_0 = 2
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = [var_1, var_2, var_3, var_2]
    var_5 = module_0.drop(var_0, var_4)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = list(var_5)
    var_7 = 0
    var_8 = 1
    var_9 = 3
    var_10 = [var_8, var_0, var_9]
    var_11 = module_0.drop(var_7, var_10)
    var_12 = list(var_11)
    var_13 = 5
    var_14 = [var_8, var_0, var_9]
    var_15 = module_0.drop(var_13, var_14)
    var_16 = list(var_15)
    var_17 = list(var_5)
    var_18 = []
    var_19 = module_0.drop(var_0, var_18)
    var_20 = list(var_19)
    var_21 = -1
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = module_0.drop(var_21, var_25)

def test_case_28():
    var_0 = 0
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_29():
    var_0 = 6
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_30():
    var_0 = 4
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_31():
    var_0 = 39
    var_1 = 6
    var_2 = [var_0, var_1, var_1, var_1, var_1, var_1]
    var_3 = None
    var_4 = module_0.split_by(var_2, criterion=var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_5 = list(var_4)

def test_case_32():
    var_0 = 2
    var_1 = None
    var_2 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_33():
    var_0 = 4
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_34():
    var_0 = -1
    var_1 = 1374
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_35():
    var_0 = -8
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_2 = list(var_1)

def test_case_36():
    var_0 = 14
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_37():
    var_0 = 1
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_38():
    var_0 = 1477
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_39():
    var_0 = -25
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__getitem__(var_0)
    assert var_3 == 600
    var_4 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    module_0.scanr(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 4
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = list(var_1)
    module_0.scanr(var_3, var_4)

def test_case_42():
    var_0 = -25
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__len__()
    assert var_3 == 0
    var_4 = var_2.__getitem__(var_3)
    assert var_4 == -25

def test_case_43():
    var_0 = 2287
    var_1 = None
    var_2 = module_0.scanl(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    var_4 = [var_0]
    var_5 = module_0.Range(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_5) == 2287
    var_6 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_7 = None
    var_8 = module_0.split_by(var_6, criterion=var_7, separator=var_0)
    var_9 = list(var_8)
    var_10 = var_9.__iter__()
    var_11 = var_5.__contains__(var_9)
    assert var_11 is False
    var_12 = module_0.split_by(var_11)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 966
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 965
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__next__()

def test_case_46():
    var_0 = 1869
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = module_0.split_by(var_1, criterion=var_2, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = -2083
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.drop_until(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__getitem__(var_0)

def test_case_48():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = 50
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
    var_7 = range(var_0)
    var_8 = module_0.LazyList(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_8.iter).__module__}.{type(var_8.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_8.exhausted is False
    assert var_8.list == []
    var_9 = 1
    var_10 = 4
    var_11 = var_8[var_9:var_10]
    var_12 = list(var_11)
    var_13 = 3
    var_14 = var_8[:var_13]
    var_15 = list(var_14)
    var_16 = 7
    var_17 = var_8[var_16:]
    var_18 = list(var_17)
    var_19 = 2
    var_20 = var_8[::var_19]
    var_21 = list(var_20)
    var_22 = 10
    with pytest.raises(IndexError):
        var_23 = var_6[var_22]