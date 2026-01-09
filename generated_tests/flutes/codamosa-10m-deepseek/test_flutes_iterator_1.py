# Check out: https://github.com/GlowCheese/deepmosa
import flutes.iterator as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
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
    var_3.index(var_0, var_0)

def test_case_1():
    var_0 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
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
    var_4 = var_3.__next__()
    assert var_4 == b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    assert var_2.list == [b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&']
    assert var_3.index == 1
    var_5 = var_2.__reversed__()

@pytest.mark.xfail(strict=True)
def test_case_2():
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
def test_case_3():
    var_0 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
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
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\xd1)\xa7\tz\xaf\xaa\xe0&'
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
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'gmN,6~nYR*z2%tw,?Pr'
    var_1 = []
    module_0.scanr(var_0, var_0, *var_1)

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
    var_2 = var_1.__reversed__()
    var_3 = var_2.__iter__()
    var_4 = module_0.drop_until(var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0\xb2'
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
    var_4 = var_3.count(var_0)
    assert var_4 == 0
    assert len(var_3) == 4
    var_4.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0\xb2'
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
    var_4 = var_3.count(var_0)
    assert var_4 == 0
    assert len(var_3) == 4
    var_3.index(var_4, stop=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 182
    var_1 = b'\x07\xd1)\xd0d\xa7\tz\xaf\xaa\xe0&'
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.chunk(var_0, var_2)
    module_0.scanr(var_3, var_3, *var_4)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = 1851
    var_3 = module_0.take(var_2, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_5 = [var_4, var_4, var_4, var_4]
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = var_6.count(var_1)
    assert var_7 == 0
    assert len(var_6) == 4
    var_8 = module_0.MapList(var_7, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.MapList'
    assert var_8.func == 0
    assert f'{type(var_8.list).__module__}.{type(var_8.list).__qualname__}' == 'builtins.generator'
    var_9 = var_8.__iter__()
    var_9.__len__()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
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
    var_4 = var_3.count(var_0)
    assert var_4 == 0
    assert len(var_3) == 4
    var_5 = var_3.__contains__(var_0)
    assert var_5 is False
    var_6 = var_3.__reversed__()
    module_0.scanr(var_3, var_3, *var_6)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_2 = module_0.LazyList(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.count(var_0)
    assert var_3 == 0
    assert len(var_2) == 10
    var_4 = var_2.__reversed__()
    module_0.scanr(var_2, var_2, *var_4)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.MapList(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.MapList'
    assert var_1.func is None
    assert var_1.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_1.index(var_0, stop=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
def test_case_16():
    var_0 = None
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_2 = [var_1, var_1, var_1, var_1]
    module_0.scanr(var_0, var_2, *var_2)

def test_case_17():
    var_0 = 0
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_0 == var_0
    with pytest.raises(TypeError):
        var_3 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = 1837
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_4 = [var_3, var_3, var_3]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_5.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_6.index == 0
    var_7 = module_0.chunk(var_1, var_6)
    module_0.scanr(var_5, var_5, *var_7)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = 1851
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_4 = [var_3, var_3, var_1, var_3, var_2, var_3]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_5.__getitem__(var_5)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = 1851
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_4 = [var_3]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = var_5.count(var_0)
    assert var_6 == 0
    assert len(var_5) == 1
    var_7 = module_0.chunk(var_0, var_6)
    var_8 = var_5.__contains__(var_6)
    assert var_8 is False
    module_0.Range(*var_4)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = 1851
    var_2 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_3 = [var_2, var_2, var_2]
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_4.exhausted is False
    assert var_4.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = module_0.MapList(var_1, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.MapList'
    assert var_5.func == 1851
    assert var_5.list is None
    var_5.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = 1857
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__iter__()
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.count(var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = 1851
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = b'\x07\x0f\xa2\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_4 = [var_3, var_3, var_3]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_5.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 1857
    var_1 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.drop(var_0, var_3)
    var_5 = module_0.chunk(var_0, var_4)
    module_0.scanr(var_3, var_3, *var_5)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 182
    var_1 = b'\xfc\xc9.\x8e\x86k<\x1a\xfa\xa4\x96i4\xae\xc6\xc2c'
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.LazyList(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.drop(var_0, var_3)
    module_0.scanr(var_3, var_3, *var_4)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = 1376
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
    var_4 = module_0.drop(var_1, var_3)
    var_5 = module_0.chunk(var_1, var_4)
    module_0.scanr(var_3, var_3, *var_5)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = 1837
    var_2 = module_0.take(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_4 = [var_1, var_3, var_3, var_3]
    var_5 = module_0.LazyList(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = module_0.drop(var_1, var_5)
    var_7 = module_0.take(var_1, var_5)
    module_0.scanr(var_5, var_5, *var_7)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0.scanl(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = None
    var_3 = 1428
    var_4 = module_0.take(var_3, var_2)
    var_5 = module_0.LazyList(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.generator'
    assert var_5.exhausted is False
    assert var_5.list == []
    var_6 = module_0.split_by(var_1, criterion=var_0)
    var_7 = module_0.chunk(var_3, var_6)
    module_0.scanr(var_5, var_5, *var_7)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = -1068
    var_2 = None
    var_3 = module_0.take(var_2, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = b'\x07\xd1)\xa7\tz\xaf\xaa\xe0&'
    var_5 = [var_4, var_4, var_4]
    var_6 = module_0.LazyList(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_6.iter).__module__}.{type(var_6.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_6.exhausted is False
    assert var_6.list == []
    var_7 = var_6.__iter__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_7.index == 0
    var_8 = module_0.chunk(var_1, var_7)
    module_0.scanr(var_6, var_6, *var_8)

def test_case_30():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = range(var_0)
    var_3 = module_0.drop_until(var_1, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(NameError):
        var_4 = list(var_3)

def test_case_31():
    var_0 = 0
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = range(var_1)
    var_6 = list(var_2)
    var_7 = 10
    var_8 = module_0.drop(var_7, var_5)
    var_9 = list(var_8)
    var_10 = range(var_1)
    var_11 = module_0.drop(var_1, var_10)
    var_12 = list(var_11)
    var_13 = -1
    var_14 = 5
    var_15 = range(var_14)
    var_16 = module_0.drop(var_13, var_15)
    with pytest.raises(ValueError):
        var_17 = list(var_16)

def test_case_32():
    var_0 = -45
    var_1 = range(var_0)
    var_2 = lambda x: x % var_0 == var_1
    var_3 = module_0.split_by(var_1, criterion=var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = ' Split by: '
    var_6 = True
    var_7 = module_0.split_by(var_5, var_6, separator=var_5)
    var_8 = list(var_7)
    var_9 = module_0.split_by(var_4, separator=var_5)
    var_10 = list(var_9)
    var_11 = 'All tests passed for split_by.'
    var_12 = print(var_11)

def test_case_33():
    var_0 = ' Split by: '
    var_1 = False
    var_2 = ''
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = list(var_3)
    var_5 = 'a.b.c'
    var_6 = var_4.__reversed__()
    var_7 = list(var_6)
    var_8 = print(var_5)

def test_case_34():
    var_0 = True
    var_1 = '.'
    var_2 = module_0.split_by(var_1, var_0, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = list(var_2)
    var_4 = 'a.b.c'
    var_5 = module_0.split_by(var_4, separator=var_1)
    var_6 = 'All tests passed for split_by.'
    var_7 = print(var_6)

def test_case_35():
    var_0 = 3
    var_1 = module_0.split_by(var_0, criterion=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = ' Split by: '
    var_3 = True
    var_4 = '.'
    var_5 = module_0.split_by(var_2, var_3, separator=var_4)
    var_6 = 'a.b.c'
    var_7 = module_0.split_by(var_6, separator=var_4)
    var_8 = list(var_7)
    var_9 = "jG4=^in*A6g/T0'0Q&2"
    var_10 = print(var_9)

def test_case_36():
    var_0 = -45
    var_1 = range(var_0)
    var_2 = 0
    var_3 = lambda x: x % var_0 == var_2
    var_4 = module_0.split_by(var_1, criterion=var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = ' Split by: '
    var_7 = True
    var_8 = '.'
    var_9 = module_0.split_by(var_6, var_7, separator=var_8)
    var_10 = list(var_9)
    var_11 = '{'
    var_12 = module_0.split_by(var_11, separator=var_8)
    var_13 = list(var_12)
    var_14 = 'All tests passed for split_by.'
    var_15 = module_0.scanr(var_11, var_10)
    var_16 = print(var_14)

def test_case_37():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = var_1.__len__()
    assert var_2 == 10
    var_3 = list(var_1)
    var_4 = var_3.__len__()
    assert var_4 == 10
    var_5 = module_0.LazyList(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_5.iter).__module__}.{type(var_5.iter).__qualname__}' == 'builtins.list_iterator'
    assert var_5.exhausted is False
    assert var_5.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_6 = False
    var_7 = '.'
    var_8 = module_0.split_by(var_7, var_6, separator=var_7)
    var_9 = list(var_8)
    var_10 = var_3.__len__()
    assert var_10 == 10
    var_11 = 'a.b.c'
    var_12 = module_0.split_by(var_11, separator=var_7)
    var_13 = list(var_12)
    var_14 = 'All tests passed for split_by.'
    var_15 = print(var_14)

def test_case_38():
    var_0 = -22
    var_1 = module_0.take(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = lambda x: x % var_0 == var_0
    var_3 = module_0.split_by(var_1, var_2, separator=var_2)
    with pytest.raises(ValueError):
        var_4 = list(var_3)

def test_case_39():
    var_0 = ''
    var_1 = module_0.drop_until(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = list(var_1)
    var_3 = '^QG91'
    var_4 = module_0.drop_until(var_2, var_3)
    with pytest.raises(TypeError):
        var_5 = list(var_4)