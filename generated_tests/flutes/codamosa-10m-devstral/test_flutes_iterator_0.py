# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.iterator as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = False
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_2.index(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = b'k'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.Range(*var_1)
    assert len(var_1) == 1
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 107
    var_3 = var_2.__len__()
    assert var_3 == 107
    var_4 = var_2.__getitem__(var_3)
    assert var_4 == 107
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.scanr(var_0, var_0)

def test_case_4():
    var_0 = b'-K\xeb\xc1\xdb\xf0\x18\x16}\x0c\xd8\xcb\xe9\xbf\xbe8\x85\xc7'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'

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
    var_1.__len__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = b'"\xea\x17\xbd\x17c\xac\x9fyD=\x10'
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
    assert var_3 == 34
    assert var_1.list == [34]
    assert var_2.index == 1
    var_4 = module_0.chunk(var_3, var_0)
    var_5 = var_4.__next__()
    var_6 = None
    var_7 = module_0.scanl(var_6, var_2, *var_2)
    assert len(var_1) == 12
    assert var_2.index == 12
    var_8 = module_0.MapList(var_2, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.MapList'
    assert f'{type(var_8.func).__module__}.{type(var_8.func).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_8.list is None
    var_8.__iter__()

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
    var_0 = b'"\xea\x17\xbd\x17c\xac\x9fyD=\x10'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.count(var_1)
    assert var_2 == 0
    assert len(var_1) == 12
    var_3 = module_0.drop_until(var_2, var_1)
    var_4 = var_1.__iter__()
    var_5 = var_4.__next__()
    assert var_5 == 34
    var_6 = module_0.chunk(var_5, var_0)
    var_7 = var_6.__next__()
    module_0.Range(*var_4)

def test_case_9():
    var_0 = b'\x12k'
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
    var_3 = module_0.scanl(var_2, var_2)
    var_4 = var_3.__next__()
    assert var_4 == 18
    assert var_1.list == [18]
    assert var_2.index == 1
    var_5 = module_0.chunk(var_4, var_0)
    var_6 = var_5.__next__()
    var_7 = {var_0, var_5, var_2, var_1}
    var_8 = module_0.scanl(var_7, var_2)
    var_9 = var_2.__next__()
    assert var_9 == 107
    assert var_1.list == [18, 107]
    assert var_2.index == 2
    var_10 = var_8.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_3 = module_0.scanr(var_0, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    module_0.Range(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
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
    var_3 = module_0.scanl(var_2, var_2, *var_2)
    assert len(var_1) == 10
    assert var_2.index == 10
    var_3.__next__()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = b'k'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = module_0.Range(*var_2)
    assert len(var_2) == 1
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 107
    var_4 = var_3.__len__()
    assert var_4 == 107
    var_5 = var_1.__next__()
    var_6 = var_3.__getitem__(var_4)
    assert var_6 == 107
    var_7 = var_3.__iter__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_7) == 107
    module_0.scanr(var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = True
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0.Range(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_2) == 0
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = None
    var_2.index(var_3, var_3)

def test_case_14():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    assert len(var_1) == 10

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 256
    var_1 = None
    var_2 = module_0.drop(var_0, var_1)
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
def test_case_16():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    assert len(var_1) == 10
    var_3 = [var_2]
    var_4 = module_0.scanl(var_3, var_3, *var_3)
    var_5 = var_4.__next__()
    assert var_5 is False
    var_2.__iter__()

def test_case_17():
    var_0 = b'\xa1HT\xb5c\x18\x8b'
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
    assert var_3 == 161
    assert var_1.list == [161]
    assert var_2.index == 1
    var_4 = module_0.chunk(var_3, var_0)
    var_5 = var_4.__next__()
    with pytest.raises(TypeError):
        var_1.__len__()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = False
    var_1 = module_0.chunk(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_1.__next__()

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = b'-K\xeb\xc1\xdb\xf0\x18\x16}\x0c\xd8\xcb\xe9\xbf\xbe8\x85\xc7'
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

def test_case_20():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    assert len(var_1) == 10
    var_3 = var_1.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = None
    var_2 = module_0.MapList(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.MapList'
    assert var_2.func is None
    assert var_2.list is None
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.index(var_1, stop=var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = True
    var_2 = None
    var_3 = module_0.take(var_1, var_2)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.LazyList(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_4.iter).__module__}.{type(var_4.iter).__qualname__}' == 'builtins.generator'
    assert var_4.exhausted is False
    assert var_4.list == []
    var_4.__contains__(var_0)

def test_case_23():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
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
    assert len(var_1) == 10
    var_4 = var_1.__len__()
    assert var_4 == 10
    var_5 = module_0.chunk(var_4, var_0)
    var_6 = var_5.__next__()
    var_7 = var_1.__iter__()

def test_case_24():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_7 = list(var_6)
    var_8 = [var_1, var_2, var_3, var_4]
    var_9 = module_0.drop(var_4, var_8)
    var_10 = list(var_9)
    var_11 = [var_1, var_2, var_3, var_4]
    var_12 = module_0.drop(var_2, var_11)
    var_13 = list(var_12)
    var_14 = 10
    var_15 = [var_1, var_2, var_3, var_4]
    var_16 = module_0.drop(var_14, var_15)
    var_17 = list(var_16)
    var_18 = 5
    var_19 = []
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

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\xd5'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    assert len(var_1) == 10
    var_3 = var_1.__getitem__(var_2)
    assert var_3 == 30
    var_4 = module_0.drop(var_2, var_2)
    var_4.__next__()

def test_case_26():
    var_0 = b'\x93\x18\xab\xd0\xcf\x978\xd5\xfd\x12'
    var_1 = var_0.__contains__(var_0)
    assert var_1 is True
    var_2 = module_0.chunk(var_1, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = var_2.__next__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = b'"\xea\x17\xbd\x17c\xac\x9fyD=\x10'
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
    assert var_3 == 34
    assert var_1.list == [34]
    assert var_2.index == 1
    var_4 = var_1.__contains__(var_1)
    assert var_4 is False
    assert len(var_1) == 12
    var_5 = module_0.chunk(var_4, var_0)
    var_5.__next__()

def test_case_28():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\x978\x13'
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    assert len(var_1) == 10
    var_3 = module_0.take(var_2, var_1)
    with pytest.raises(StopIteration):
        var_3.__next__()

def test_case_29():
    var_0 = b'"\xea\x17\xbd\x17c\xac\x9fyD=\x10'
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
    assert var_3 == 34
    assert var_1.list == [34]
    assert var_2.index == 1
    var_4 = module_0.chunk(var_3, var_0)
    var_5 = var_4.__next__()
    var_6 = var_1.__iter__()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'flutes.iterator.LazyList.LazyListIterator'
    assert var_6.index == 0

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = b'\x1e\x93\x18\xab\xd0\xbd\xcf\xb7\x978>'
    var_1 = None
    var_2 = module_0.split_by(var_0, var_1, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2.__next__()

def test_case_31():
    var_0 = b''
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    with pytest.raises(StopIteration):
        var_1.__next__()

def test_case_32():
    var_0 = b'k'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__next__()

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = b'\x93\x18\xab\xd0\xcf\x978\xd5\xfd\x12'
    var_1 = module_0.drop_until(var_0, var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_1.__next__()

def test_case_34():
    var_0 = b't\xbd1\xf4\x05v\x1d\xe78\xf5Jo\xb1'
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
    var_3 = None
    var_4 = var_1.__contains__(var_3)
    assert var_4 is False
    assert len(var_1) == 13
    var_5 = var_1.__len__()
    assert var_5 == 13
    var_6 = var_1.__reversed__()
    var_7 = module_0.take(var_5, var_2)
    var_8 = var_7.__next__()
    assert var_8 == 177
    var_9 = var_1.__iter__()

def test_case_35():
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
    var_3 = -11
    with pytest.raises(IndexError):
        var_4 = var_2[var_3]

def test_case_36():
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
    var_3 = 10
    with pytest.raises(IndexError):
        var_4 = var_2[var_3]

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = b'k'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = var_2.__contains__(var_1)
    assert var_3 is False
    assert len(var_2) == 1
    var_4 = module_0.Range(*var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_4) == 107
    var_5 = var_4.__len__()
    assert var_5 == 107
    var_6 = var_1.__next__()
    var_7 = var_4.__getitem__(var_5)
    assert var_7 == 107
    var_8 = var_4.__iter__()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_8) == 107
    module_0.scanr(var_3, var_7, *var_8)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = b'k'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = module_0.Range(*var_2)
    assert len(var_2) == 1
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 107
    var_4 = var_3.__len__()
    assert var_4 == 107
    var_5 = var_1.__next__()
    var_6 = var_3.__getitem__(var_4)
    assert var_6 == 107
    var_7 = var_3.__next__()
    assert var_7 == 0
    var_6.__next__()

def test_case_39():
    var_0 = b'k'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = module_0.Range(*var_2)
    assert len(var_2) == 1
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_3) == 107
    var_4 = var_3.__len__()
    assert var_4 == 107
    var_5 = var_1.__next__()
    var_6 = var_3.__getitem__(var_4)
    assert var_6 == 107
    var_7 = var_3.__iter__()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_7) == 107
    var_8 = module_0.LazyList(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_8.iter).__module__}.{type(var_8.iter).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_8.iter) == 107
    assert var_8.exhausted is False
    assert var_8.list == []

def test_case_40():
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
    var_8 = module_0.take(var_4, var_7)
    var_9 = list(var_8)
    var_10 = 0
    var_11 = [var_1, var_2, var_0]
    var_12 = module_0.take(var_10, var_11)
    var_13 = list(var_12)
    var_14 = []
    var_15 = module_0.take(var_4, var_14)
    var_16 = list(var_15)
    var_17 = 10
    var_18 = range(var_17)
    var_19 = -1
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = module_0.take(var_19, var_23)
    with pytest.raises(ValueError):
        var_25 = list(var_24)

def test_case_41():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_0, var_1, var_2]
    var_4 = module_0.drop(var_0, var_3)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_5 = list(var_4)
    var_6 = 10
    var_7 = 5
    var_8 = []
    var_9 = module_0.drop(var_7, var_8)
    var_10 = list(var_9)
    var_11 = -1
    var_12 = 1
    var_13 = 2
    var_14 = 3
    var_15 = [var_12, var_13, var_14]
    var_16 = module_0.drop(var_11, var_15)
    var_17 = range(var_6)
    var_18 = 'hello'
    var_19 = module_0.drop(var_14, var_18)
    var_20 = list(var_19)

def test_case_42():
    var_0 = b''
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.drop_until(var_1, var_1)
    with pytest.raises(StopIteration):
        var_2.__next__()

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = b'k3'
    var_1 = module_0.split_by(var_0, var_0, separator=var_0)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.LazyList(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_2.iter).__module__}.{type(var_2.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_2.exhausted is False
    assert var_2.list == []
    var_3 = var_2.__contains__(var_1)
    assert var_3 is False
    assert len(var_2) == 2
    var_4 = module_0.Range(*var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'flutes.iterator.Range'
    assert var_4.l == 107
    assert var_4.r == 51
    assert var_4.step == 1
    assert var_4.val == 107
    assert var_4.length == -56
    var_5 = var_4.__len__()
    assert var_5 == -56
    var_6 = var_1.__next__()
    var_7 = var_4.__getitem__(var_5)
    assert var_7 == -5
    module_0.Range()

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = b'k'
    var_1 = None
    var_2 = module_0.split_by(var_0, var_0, criterion=var_0, separator=var_1)
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_3 = module_0.LazyList(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_3.iter).__module__}.{type(var_3.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_3.exhausted is False
    assert var_3.list == []
    var_4 = var_3.__reversed__()
    var_5 = var_4.__iter__()
    var_6 = var_3.__contains__(var_4)
    assert var_6 is False
    assert len(var_3) == 1
    var_7 = module_0.Range(*var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'flutes.iterator.Range'
    assert len(var_7) == 107
    var_8 = var_7.__len__()
    assert var_8 == 107
    var_2.__next__()

def test_case_45():
    var_0 = b''
    var_1 = module_0.LazyList(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.iterator.LazyList'
    assert f'{type(var_1.iter).__module__}.{type(var_1.iter).__qualname__}' == 'builtins.bytes_iterator'
    assert var_1.exhausted is False
    assert var_1.list == []
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.A).__module__}.{type(module_0.A).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.B).__module__}.{type(module_0.B).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.R).__module__}.{type(module_0.R).__qualname__}' == 'typing.TypeVar'
    var_2 = 237
    var_3 = module_0.chunk(var_2, var_1)
    with pytest.raises(StopIteration):
        var_3.__next__()