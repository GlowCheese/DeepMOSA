# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_0.scanr(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.LazyList(var_1)

def test_case_3():
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
    var_1.__len__()

def test_case_5():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 1
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = next(var_3)
    assert var_4 == 5

def test_case_6():
    var_0 = 1
    var_1 = 12
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 5
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4[-1]
    assert var_5 == 9
    var_6 = var_4[-3]
    assert var_6 == 5

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = [var_0]
    module_0.scanr(var_1, var_1, *var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'd@=^.'
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
    assert len(var_1) == 5
    var_4 = None
    module_0.scanr(var_4, var_1)

def test_case_10():
    var_0 = 'd@=^.'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    assert len(var_1) == 5

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
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_3.index == 0
    var_4 = module_0.take(var_1, var_2)
    var_5 = module_0.take(var_2, var_2)
    var_2.__len__()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = [var_2]
    var_4 = module_0.scanr(var_2, var_3)
    var_5 = module_0.split_by(var_4, criterion=var_0)
    var_1.index(var_4, stop=var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = [var_2]
    var_4 = module_0.scanr(var_2, var_3)
    var_5 = var_4.__iter__()
    var_1.__getitem__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = None
    var_2 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__contains__(var_1)

def test_case_15():
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
    var_7 = var_6[-3]
    assert var_7 == 3

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'd@=^.'
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
    var_1.__getitem__(var_2)

def test_case_17():
    var_0 = 'd@=^.'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    assert len(var_1) == 5
    var_3 = var_1.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'd@=^.'
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
    assert len(var_1) == 5
    var_4 = var_1.__len__()
    assert var_4 == 5
    var_5 = var_1.__iter__()
    var_6 = var_1.__contains__(var_0)
    assert var_6 is False
    var_7 = True
    var_6.__contains__(var_7)

def test_case_19():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3, var_1]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = var_5[-1]
    var_7 = var_5[-3]
    assert var_7 == 3

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'd@=^.'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.str_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__reversed__()
    var_3 = module_0.MapList(var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_3.func).__module__}.{type(var_3.func).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.list).__module__}.{type(var_3.list).__qualname__}' == 'flutes.iterator.LazyList'
    var_4 = var_1.__iter__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.index == 0
    var_5 = var_4.__iter__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_5.index == 0
    var_6 = module_0.drop_until(var_3, var_1)
    var_7 = var_1.__reversed__()
    var_7.__len__()

def test_case_21():
    var_0 = '\rrs0WaF('
    var_1 = module_0.split_by(var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)

def test_case_22():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.split_by(var_1, criterion=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_23():
    var_0 = ' Split by: '
    var_1 = False
    var_2 = module_0.chunk(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_24():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = False
    var_1 = None
    var_2 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_3.__contains__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = True
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 1
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_4.__getitem__(var_0)

def test_case_27():
    var_0 = True
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_2 = list(var_1)

def test_case_28():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'o'])
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = True
    var_1 = None
    var_2 = module_0.drop(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.generator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = False
    var_1 = None
    var_2 = None
    var_3 = module_0.split_by(var_1, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = [var_2, var_2, var_2, var_1]
    module_0.scanr(var_2, var_4, *var_4)

def test_case_31():
    var_0 = -1
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_2 = list(var_1)

def test_case_32():
    var_0 = 2
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)

def test_case_33():
    var_0 = -1558
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(ValueError):
        var_3 = list(var_2)

def test_case_34():
    var_0 = 2
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_35():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = bool(var_4[2:2] == [])
    assert var_5 is True
    var_6 = var_4[5:10]

def test_case_36():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4[var_1]
    var_6 = var_4[0]
    assert var_6 == 1
    var_7 = var_4[1]
    assert var_7 == 2
    var_8 = var_4[2]
    assert var_8 == 3

def test_case_37():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 1
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = next(var_3)
    assert var_4 == 5
    with pytest.raises(StopIteration):
        var_5 = next(var_3)

def test_case_38():
    var_0 = 'Split by: '
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_39():
    var_0 = True
    var_1 = ' '
    var_2 = module_0.split_by(var_1, var_0, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = bool(var_3 == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']])

def test_case_40():
    var_0 = ' Split by: '
    var_1 = False
    var_2 = ' '
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = bool(var_2 == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']])

def test_case_41():
    var_0 = True
    var_1 = ' '
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)

def test_case_42():
    var_0 = 3
    var_1 = [var_0]
    var_2 = module_0.chunk(var_0, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = bool(var_1 == [])

def test_case_43():
    var_0 = -29
    var_1 = [var_0]
    var_2 = module_0.drop_until(var_1, var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(TypeError):
        var_3 = list(var_2)

def test_case_44():
    var_0 = []
    var_1 = module_0.drop_until(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)

def test_case_45():
    var_0 = 1
    var_1 = 33
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.Range(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 32
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = var_3[1:8:2]
    var_5 = bool(var_2[1:8:2] == [2, 4, 6, 8])
    var_6 = var_3[::3]
    var_7 = bool(var_3[::3] == [1, 4, 7, 10])

def test_case_46():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 5
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[2]
    assert var_6 == 5
    var_7 = var_4[4]
    assert var_7 == 9

def test_case_47():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.MapList'
    assert len(var_7) == 5
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_8 = var_7[1:4]
    var_9 = bool(var_7[1:4] == [4, 9, 16])
    assert var_9 is True
    var_10 = var_7[:3]
    var_11 = bool(var_7[:3] == [1, 4, 9])
    assert var_11 is True
    var_12 = var_7[2:]
    var_13 = bool(var_7[2:] == [9, 16, 25])
    assert var_13 is True
    var_14 = var_7[::2]
    var_15 = bool(var_7[::2] == [1, 9, 25])
    assert var_15 is True
    var_16 = var_7[1:4:2]
    var_17 = bool(var_7[1:4:2] == [4, 16])
    assert var_17 is True