# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.scanl(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.MapList(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert var_2.func is None
    assert f'{type(var_2.list).__module__}.{type(var_2.list).__qualname__}' == 'builtins.generator'
    var_2.index(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.scanr(var_0, var_0)

def test_case_3():
    var_0 = -4501
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reversed__()
    var_2.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = -4501
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
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
    var_5 = var_4.count(var_1)
    assert var_5 == 0
    assert len(var_4) == 0
    var_6 = module_0.MapList(var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_6) == 0
    var_7 = var_6.count(var_5)
    assert var_7 == 0
    var_5.__len__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = ':?OyU\x0cK\t'
    var_2 = module_0.MapList(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert var_2.func == ':?OyU\x0cK\t'
    assert var_2.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__len__()

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
    var_0 = False
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
    var_2.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.scanl(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_0, var_0, criterion=var_0)
    var_3 = [var_0]
    module_0.Range(*var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.split_by(var_0, var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0.scanr(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -4501
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
    var_2.__contains__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_0, var_0, criterion=var_0)
    var_3 = None
    var_4 = [var_0, var_3, var_3, var_0]
    module_0.Range(*var_4)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 3011
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_1.__iter__()
    var_4 = list(var_2)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_5.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_6.index == 0
    var_7 = var_6.__iter__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_7.index == 0
    var_8 = var_4.__len__()
    assert var_8 == 1
    var_9 = var_5.count(var_1)
    assert var_9 == 0
    assert len(var_5) == 1
    var_8.__contains__(var_9)

def test_case_15():
    var_0 = -4472
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
    with pytest.raises(TypeError):
        var_2.__len__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = -4501
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_1.count(var_0)
    assert var_4 == 0
    var_5 = None
    var_3.__getitem__(var_5)

def test_case_17():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.scanl(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_1, var_1, criterion=var_1)
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__contains__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = None
    var_2 = [var_1]
    var_3 = [var_0, var_1, var_0, var_0]
    module_0.scanr(var_1, var_2, *var_3)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = [var_0]
    module_0.scanr(var_0, var_1, *var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = -4542
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.generator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_2.__contains__(var_2)

def test_case_22():
    var_0 = 4
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = var_1.__iter__()
    with pytest.raises(TypeError):
        var_4 = list(var_2)

def test_case_23():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_24():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 5
    var_6 = 1
    var_7 = var_4.__contains__(var_2)
    assert var_7 is False
    var_8 = 2
    var_9 = [var_6, var_8, var_0]
    var_10 = module_0.chunk(var_5, var_9)
    var_11 = list(var_10)
    var_12 = [var_6, var_8, var_0]
    var_13 = module_0.chunk(var_0, var_12)
    var_14 = list(var_13)
    var_15 = [var_6, var_8, var_0]
    var_16 = module_0.chunk(var_6, var_15)
    var_17 = list(var_16)
    var_18 = module_0.take(var_5, var_17)
    var_19 = list(var_18)
    var_20 = 4
    var_21 = [var_6, var_8, var_0, var_20]
    var_22 = iter(var_21)
    var_23 = module_0.chunk(var_8, var_22)
    var_24 = list(var_23)
    var_25 = var_17.__iter__()
    var_26 = list(var_25)
    var_27 = range(var_5)
    var_28 = 0
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = module_0.chunk(var_28, var_32)
    with pytest.raises(ValueError):
        var_34 = list(var_33)

def test_case_25():
    var_0 = -14
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = -10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_3.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = -4501
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
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
    var_5 = var_4.count(var_1)
    assert var_5 == 0
    assert len(var_4) == 0
    var_6 = var_1.__reversed__()
    var_4.__getitem__(var_6)

def test_case_28():
    var_0 = -4501
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.split_by(var_1, separator=var_1)
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_29():
    var_0 = -4501
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = var_3.count(var_4)
    assert var_5 == 0

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 3011
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
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
    assert len(var_3) == 3011
    var_2.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = -4501
    var_1 = range(var_0)
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.count(var_1)
    assert var_4 == 0
    assert len(var_3) == 0
    var_5 = var_3.__len__()
    assert var_5 == 0
    var_6 = var_3.count(var_5)
    assert var_6 == 0
    var_7 = None
    var_4.__getitem__(var_7)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 2987
    var_1 = range(var_0)
    var_2 = module_0.drop(var_0, var_1)
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
    var_5 = var_4.count(var_1)
    assert var_5 == 0
    assert len(var_4) == 0
    var_4.index(var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 2984
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
    var_2.count(var_1)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 2966
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_1.__iter__()
    var_4 = list(var_2)
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_1.__len__()
    assert var_6 == 2966
    var_7 = var_1.count(var_6)
    assert var_7 == 0
    var_5.__getitem__(var_6)

def test_case_35():
    var_0 = 4
    var_1 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__iter__()
    with pytest.raises(TypeError):
        var_3 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'hello world'
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_1)
    var_4 = ' Split by: '
    var_5 = True
    var_6 = module_0.split_by(var_4, var_5, separator=var_1)
    var_7 = list(var_6)
    var_7.__getitem__(var_7)

def test_case_37():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_0, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = 0
    var_9 = module_0.drop(var_8, var_5)
    var_10 = list(var_9)
    var_11 = 10
    var_12 = [var_1, var_0, var_2]
    var_13 = module_0.drop(var_11, var_12)
    var_14 = list(var_13)
    var_15 = []
    var_16 = module_0.drop(var_2, var_15)
    var_17 = list(var_16)
    var_18 = range(var_4)
    var_19 = 'hello'
    var_20 = module_0.drop(var_0, var_19)
    var_21 = list(var_20)
    var_22 = range(var_11)
    var_23 = module_0.drop(var_4, var_22)
    var_24 = list(var_23)
    var_25 = -1
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = module_0.drop(var_25, var_29)
    with pytest.raises(ValueError):
        var_31 = list(var_30)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 4
    var_1 = range(var_0)
    var_2 = True
    var_3 = module_0.drop(var_2, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.split_by(var_1, separator=var_1)
    var_5 = var_1.__iter__()
    var_6 = list(var_4)
    var_7 = module_0.LazyList(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_7.iter).__module__}.{type(var_7.iter).__qualname__}' == 'builtins.generator'
    assert var_7.exhausted is False
    assert var_7.list == []
    var_8 = module_0.split_by(var_5)
    var_9 = module_0.chunk(var_2, var_6)
    var_10 = var_1.count(var_6)
    assert var_10 == 0
    var_11 = var_6.__iter__()
    var_12 = None
    var_13 = var_1.__contains__(var_12)
    assert var_13 is False
    var_14 = var_7.__reversed__()
    var_15 = module_0.scanl(var_12, var_14)
    var_7.__getitem__(var_12)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 4
    var_1 = range(var_0)
    var_2 = True
    var_3 = module_0.drop(var_2, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.split_by(var_1, separator=var_1)
    var_5 = var_1.__iter__()
    var_6 = list(var_4)
    var_7 = module_0.LazyList(var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_7.iter).__module__}.{type(var_7.iter).__qualname__}' == 'builtins.range_iterator'
    assert var_7.exhausted is False
    assert var_7.list == []
    var_8 = False
    var_9 = module_0.split_by(var_5, var_8, criterion=var_6)
    var_10 = -2048
    var_11 = module_0.chunk(var_10, var_6)
    var_12 = var_6.count(var_6)
    assert var_12 == 0
    var_13 = var_7.__contains__(var_8)
    assert var_13 is True
    assert var_7.list == [0]
    var_14 = var_7.count(var_1)
    assert var_14 == 0
    assert len(var_7) == 4
    var_15 = module_0.scanl(var_13, var_1)
    var_14.count(var_6)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 2
    var_1 = range(var_0)
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.split_by(var_1, separator=var_1)
    var_4 = var_1.__iter__()
    var_5 = list(var_3)
    var_6 = module_0.LazyList(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.generator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = module_0.split_by(var_5)
    var_8 = module_0.chunk(var_5, var_1)
    var_9 = var_6.count(var_5)
    assert var_9 == 0
    assert len(var_6) == 2
    var_10 = var_6.count(var_1)
    assert var_10 == 0
    var_11 = var_5.count(var_9)
    assert var_11 == 0
    var_4.count(var_9)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = False
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
    var_4 = var_2.__getitem__(var_3)
    assert var_4 == 0
    module_0.LazyList(var_4)

def test_case_42():
    var_0 = False
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
    var_4 = var_2.__getitem__(var_3)
    assert var_4 == 0
    var_5 = var_2.count(var_4)
    assert var_5 == 0
    var_6 = var_2.count(var_2)
    assert var_6 == 0

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 4
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = module_0.drop(var_0, var_2)
    var_4 = None
    var_5 = [var_0]
    var_6 = module_0.Range(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_6) == 4
    var_7 = var_6.__contains__(var_4)
    assert var_7 is False
    var_7.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = -2297
    var_1 = lambda x: x % var_0 == var_0
    var_2 = 'hello world'
    var_3 = ' '
    var_4 = module_0.split_by(var_2, separator=var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = ' Split by: '
    var_7 = False
    var_8 = module_0.split_by(var_6, var_7, separator=var_3)
    var_9 = list(var_8)
    var_10 = None
    var_11 = module_0.split_by(var_9, criterion=var_10)
    var_1.count(var_1)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = -1
    var_1 = module_0.take(var_0, var_0)
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
    var_5 = True
    var_6 = None
    var_7 = module_0.chunk(var_6, var_6)
    var_8 = module_0.chunk(var_5, var_6)
    var_4.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = -1
    var_1 = module_0.take(var_0, var_0)
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
    var_5 = 1478.159
    var_6 = [var_5, var_2, var_5]
    var_7 = module_0.Range(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.Range'
    assert var_7.l == pytest.approx(1478.159, abs=0.01, rel=0.01)
    assert var_7.r is True
    assert var_7.step == pytest.approx(1478.159, abs=0.01, rel=0.01)
    assert var_7.val == pytest.approx(1478.159, abs=0.01, rel=0.01)
    assert var_7.length == pytest.approx(-1.0, abs=0.01, rel=0.01)
    var_8 = var_7.__len__()
    assert var_8 == pytest.approx(-1.0, abs=0.01, rel=0.01)
    var_9 = var_7.__getitem__(var_8)
    assert var_9 == pytest.approx(-1478.159, abs=0.01, rel=0.01)
    var_10 = module_0.drop(var_8, var_9)
    module_0.LazyList(var_8)