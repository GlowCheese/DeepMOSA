####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[2:5] == [2, 3, 4]
    assert lazy_list[:3] == [0, 1, 2]
    assert lazy_list[5:] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test out of bounds access
    try:
        _ = lazy_list[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with non-sequential iterable
    lazy_list = LazyList(x for x in [1, 2, 3, 4, 5])
    assert lazy_list[0] == 1
    assert lazy_list[2:4] == [3, 4]

    # Test with exhausted list
    lazy_list = LazyList(range(3))
    _ = lazy_list[2]  # Exhaust the list
    assert lazy_list[0] == 0
    assert lazy_list[1:3] == [1, 2]


# LLM-generated content at query #2
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(0, 0, 1)) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[1:5] == [1, 2, 3, 4]
    assert lazy_list[:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]
    assert lazy_list[-1:0:-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test with non-sequential iterable
    lazy_list = LazyList(x for x in [1, 2, 3, 4, 5])
    assert lazy_list[0] == 1
    assert lazy_list[2] == 3
    assert lazy_list[1:4] == [2, 3, 4]

    # Test with exhausted iterable
    lazy_list = LazyList(range(5))
    _ = lazy_list[4]  # Exhaust the iterable
    assert lazy_list[0] == 0
    assert lazy_list[4] == 4
    assert lazy_list[:] == [0, 1, 2, 3, 4]


# LLM-generated content at query #4
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(1, 1)) == 0
    assert len(Range(1, 1, 2)) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:10] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test out of bounds access
    try:
        _ = lazy_list[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with non-sequential access
    lazy_list = LazyList(range(10))
    assert lazy_list[9] == 9  # Access last element first
    assert lazy_list[0] == 0  # Then first element
    assert lazy_list[5] == 5  # Then middle element

    # Test with exhausted iterator
    lazy_list = LazyList(range(5))
    _ = list(lazy_list)  # Exhaust the iterator
    assert lazy_list[0] == 0
    assert lazy_list[4] == 4
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]


# LLM-generated content at query #6
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(5)) == 5
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test with start and stop
    assert len(Range(1, 5)) == 4
    assert len(Range(5, 1)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(-3, 3)) == 6

    # Test with start, stop, and step
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(10, 0, 2)) == 0
    assert len(Range(0, 10, -1)) == 0
    assert len(Range(0, 10, 1)) == 10
    assert len(Range(0, 10, 10)) == 1
    assert len(Range(0, 10, 11)) == 1
    assert len(Range(0, 10, 100)) == 1


# LLM-generated content at query #7
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5

    # Test slice with positive indices
    assert r[1:5] == [1, 2, 3, 4]
    assert r[:5] == [0, 1, 2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]

    # Test slice with negative indices
    assert r[-5:-1] == [5, 6, 7, 8]
    assert r[-5:] == [5, 6, 7, 8, 9]
    assert r[:-1] == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Test with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[-1] == 9
    assert r[1:3] == [3, 5]
    assert r[::2] == [1, 5, 9]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = r[10]
    with pytest.raises(IndexError):
        _ = r[-10]


# LLM-generated content at query #8
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    r1 = Range(10)
    assert len(r1) == 10

    # Test with start and stop
    r2 = Range(1, 11)
    assert len(r2) == 10

    # Test with start, stop, and step
    r3 = Range(1, 11, 2)
    assert len(r3) == 5

    # Test with negative step (should still work as length is calculated)
    r4 = Range(10, 0, -2)
    assert len(r4) == 5

    # Test with zero length
    r5 = Range(5, 5)
    assert len(r5) == 0

    # Test with negative range (should have zero length)
    r6 = Range(-5, -10)
    assert len(r6) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_drop_until():
    # Test dropping until a condition is met
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5])) == [3, 4, 5]
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7])) == [6, 7]

    # Test when the condition is never met
    assert list(drop_until(lambda x: x > 10, range(5))) == []

    # Test when the first element meets the condition
    assert list(drop_until(lambda x: x == 1, [1, 2, 3])) == [1, 2, 3]

    # Test with an empty iterable
    assert list(drop_until(lambda x: x > 0, [])) == []

    # Test with a custom object
    class CustomObj:
        def __init__(self, val):
            self.val = val

    objs = [CustomObj(1), CustomObj(2), CustomObj(3)]
    assert list(drop_until(lambda x: x.val == 2, objs)) == [objs[1], objs[2]]


# LLM-generated content at query #10
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(1, 1)) == 0
    assert len(Range(1, 1, 2)) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5

    # Test slice
    assert r[1:5] == [1, 2, 3, 4]
    assert r[:5] == [0, 1, 2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1:8:2] == [1, 3, 5, 7]

    # Test with start, stop, step
    r = Range(1, 11, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[4] == 9
    assert r[-1] == 9
    assert r[1:4] == [3, 5, 7]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = r[10]
    with pytest.raises(IndexError):
        _ = r[-10]

    # Test empty range
    r = Range(0)
    with pytest.raises(IndexError):
        _ = r[0]
    assert r[0:0] == []


# LLM-generated content at query #12
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(0, 0, 1)) == 0


# LLM-generated content at query #13
#--------------------------

```python
def test_take():
    # Test taking elements from a list
    assert list(take(3, [1, 2, 3, 4, 5])) == [1, 2, 3]
    assert list(take(5, [1, 2, 3])) == [1, 2, 3]

    # Test taking zero elements
    assert list(take(0, [1, 2, 3])) == []

    # Test taking elements from an empty iterable
    assert list(take(5, [])) == []

    # Test taking elements from a generator
    assert list(take(3, (x for x in range(10)))) == [0, 1, 2]

    # Test negative n raises ValueError
    try:
        list(take(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_MapList___getitem__():
    # Test with integer index
    lst = [1, 2, 3, 4, 5]
    func = lambda x: x * 2
    map_list = MapList(func, lst)
    assert map_list[0] == 2
    assert map_list[1] == 4
    assert map_list[2] == 6
    assert map_list[3] == 8
    assert map_list[4] == 10

    # Test with negative index
    assert map_list[-1] == 10
    assert map_list[-2] == 8

    # Test with slice
    assert map_list[1:3] == [4, 6]
    assert map_list[:3] == [2, 4, 6]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[::2] == [2, 6, 10]

    # Test with empty slice
    assert map_list[2:2] == []

    # Test with out-of-bounds index
    try:
        _ = map_list[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with out-of-bounds negative index
    try:
        _ = map_list[-10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with out-of-bounds slice
    assert map_list[10:20] == []


# LLM-generated content at query #15
#--------------------------

```python
def test_drop():
    # Test dropping 0 elements
    assert list(drop(0, [1, 2, 3, 4])) == [1, 2, 3, 4]

    # Test dropping all elements
    assert list(drop(4, [1, 2, 3, 4])) == []

    # Test dropping some elements
    assert list(drop(2, [1, 2, 3, 4])) == [3, 4]

    # Test dropping more elements than available
    assert list(drop(10, [1, 2, 3, 4])) == []

    # Test with empty iterable
    assert list(drop(5, [])) == []

    # Test with negative n (should raise ValueError)
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with generator
    gen = (x for x in range(10))
    assert list(drop(5, gen)) == [5, 6, 7, 8, 9]

    # Test with string
    assert list(drop(3, "hello")) == ['l', 'o']


# LLM-generated content at query #16
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[1:5] == [1, 2, 3, 4]
    assert lazy_list[:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]
    assert lazy_list[-1::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test with non-sequential iterable
    lazy_list = LazyList(x for x in [1, 4, 9, 16, 25])
    assert lazy_list[0] == 1
    assert lazy_list[2] == 9
    assert lazy_list[1:4] == [4, 9, 16]

    # Test with empty slice
    assert lazy_list[5:2] == []
    assert lazy_list[10:20] == []

    # Test with exhausted iterable
    lazy_list = LazyList(range(5))
    _ = list(lazy_list)  # Exhaust the iterable
    assert lazy_list[0] == 0
    assert lazy_list[4] == 4
    assert lazy_list[1:4] == [1, 2, 3]


# LLM-generated content at query #17
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(0, 0, 1)) == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(10)) == 10
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test with two arguments (start, stop)
    assert len(Range(1, 10)) == 9
    assert len(Range(0, 0)) == 0
    assert len(Range(-5, 5)) == 10
    assert len(Range(10, 1)) == 0

    # Test with three arguments (start, stop, step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(0, 10, 1)) == 10
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(0, 10, -1)) == 0
    assert len(Range(0, 10, 100)) == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(10)) == 10
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test with two arguments (start, stop)
    assert len(Range(1, 10)) == 9
    assert len(Range(0, 0)) == 0
    assert len(Range(-5, 5)) == 10

    # Test with three arguments (start, stop, step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(0, 0, 1)) == 0
    assert len(Range(0, 0, -1)) == 0

    # Test with negative steps
    assert len(Range(5, -5, -1)) == 10
    assert len(Range(5, -5, -2)) == 5

    # Test edge cases
    assert len(Range(1, 1)) == 0
    assert len(Range(1, 2)) == 1
    assert len(Range(1, 1, 1)) == 0
    assert len(Range(1, 1, -1)) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_MapList___getitem__():
    # Test with single index
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * 2, lst)
    assert map_list[0] == 2
    assert map_list[2] == 6
    assert map_list[-1] == 10

    # Test with slice
    assert map_list[1:3] == [4, 6]
    assert map_list[:2] == [2, 4]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[::2] == [2, 6, 10]

    # Test with empty slice
    assert map_list[1:1] == []

    # Test with negative indices in slice
    assert map_list[-3:-1] == [6, 8]

    # Test with step in slice
    assert map_list[0:4:2] == [2, 6]

    # Test with out of bounds index (should raise IndexError)
    try:
        map_list[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with out of bounds slice (should return empty list)
    assert map_list[10:20] == []


# LLM-generated content at query #21
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, range(10))) == [3, 4, 5, 6, 7, 8, 9]

    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []

    # Test with all elements satisfying the predicate
    assert list(drop_until(lambda x: x >= 0, range(10))) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Test with no elements satisfying the predicate
    assert list(drop_until(lambda x: x > 100, range(10))) == []

    # Test with the first element satisfying the predicate
    assert list(drop_until(lambda x: x == 0, range(10))) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Test with the last element satisfying the predicate
    assert list(drop_until(lambda x: x == 9, range(10))) == [9]

    # Test with a custom iterable
    assert list(drop_until(lambda x: x.startswith('b'), ['apple', 'banana', 'cherry'])) == ['banana', 'cherry']


# LLM-generated content at query #22
#--------------------------

```python
def test_drop_until():
    # Test dropping until a condition is met
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5])) == [3, 4, 5]
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7])) == [6, 7]

    # Test when the condition is met at the first element
    assert list(drop_until(lambda x: x == 1, [1, 2, 3])) == [1, 2, 3]

    # Test when the condition is never met
    assert list(drop_until(lambda x: x > 10, range(5))) == []

    # Test with an empty iterable
    assert list(drop_until(lambda x: x > 0, [])) == []

    # Test with a custom object
    class CustomObj:
        def __init__(self, val):
            self.val = val

    objs = [CustomObj(1), CustomObj(2), CustomObj(3)]
    assert list(drop_until(lambda x: x.val == 2, objs)) == [objs[1], objs[2]]


# LLM-generated content at query #23
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5
    assert r[-10] == 0

    # Test slice with positive indices
    assert r[0:3] == [0, 1, 2]
    assert r[2:5] == [2, 3, 4]
    assert r[5:10] == [5, 6, 7, 8, 9]

    # Test slice with negative indices
    assert r[-3:-1] == [7, 8]
    assert r[-5:-2] == [5, 6, 7]
    assert r[-10:-5] == [0, 1, 2, 3, 4]

    # Test slice with step
    assert r[0:10:2] == [0, 2, 4, 6, 8]
    assert r[1:10:3] == [1, 4, 7]
    assert r[-1:-10:-2] == [9, 7, 5, 3, 1]

    # Test Range with start, stop, step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-3] == 5
    assert r[0:3] == [1, 3, 5]
    assert r[1:4] == [3, 5, 7]
    assert r[0:5:2] == [1, 5, 9]

    # Test out of bounds index
    with pytest.raises(IndexError):
        r[10]
    with pytest.raises(IndexError):
        r[-11]


# LLM-generated content at query #24
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:10] == [5, 6, 7, 8, 9]
    assert lazy_list[2:8:2] == [2, 4, 6]

    # Test slice with negative indices
    assert lazy_list[-5:-1] == [5, 6, 7, 8]
    assert lazy_list[-1:-5:-1] == [9, 8, 7, 6]

    # Test out of bounds access
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test empty slice
    assert lazy_list[5:5] == []
    assert lazy_list[10:15] == []

    # Test with non-sequential iterable
    lazy_list = LazyList([1, 3, 5, 7, 9])
    assert lazy_list[0] == 1
    assert lazy_list[2] == 5
    assert lazy_list[1:4] == [3, 5, 7]

    # Test that iterable is exhausted after full access
    lazy_list = LazyList(range(5))
    assert len(lazy_list.list) == 0
    _ = lazy_list[4]
    assert len(lazy_list.list) == 5
    assert lazy_list.exhausted


# LLM-generated content at query #25
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5
    assert r[-10] == 0

    # Test slice with positive indices
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[5:10] == [5, 6, 7, 8, 9]

    # Test slice with negative indices
    assert r[-5:-1] == [5, 6, 7, 8]
    assert r[-10:-5] == [0, 1, 2, 3, 4]

    # Test slice with step
    assert r[0:10:2] == [0, 2, 4, 6, 8]
    assert r[1:10:3] == [1, 4, 7]

    # Test Range with start, stop, step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-3] == 5
    assert r[0:3] == [1, 3, 5]
    assert r[1:4] == [3, 5, 7]
    assert r[0:5:2] == [1, 5, 9]

    # Test empty slice
    assert r[5:5] == []
    assert r[10:20] == []

    # Test out of bounds index
    with pytest.raises(IndexError):
        _ = r[10]
    with pytest.raises(IndexError):
        _ = r[-11]


# LLM-generated content at query #26
#--------------------------

```python
def test_drop_until():
    # Test dropping until condition is met
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5])) == [3, 4, 5]
    assert list(drop_until(lambda x: x.startswith('b'), ['a', 'b', 'c'])) == ['b', 'c']

    # Test with empty iterable
    assert list(drop_until(lambda x: x > 0, [])) == []

    # Test with all elements dropped
    assert list(drop_until(lambda x: x > 10, range(5))) == []

    # Test with first element satisfying condition
    assert list(drop_until(lambda x: x == 1, [1, 2, 3])) == [1, 2, 3]

    # Test with custom objects
    class TestObj:
        def __init__(self, val):
            self.val = val

    objs = [TestObj(1), TestObj(2), TestObj(3)]
    assert list(drop_until(lambda x: x.val > 1, objs)) == [objs[1], objs[2]]


# LLM-generated content at query #27
#--------------------------

```python
def test_LazyList___getitem__():
    # Test with single index
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test with negative index
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test with slice
    assert lazy_list[1:5] == [1, 2, 3, 4]
    assert lazy_list[:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test with exhausted list
    exhausted_list = LazyList([1, 2, 3])
    _ = list(exhausted_list)  # Exhaust the list
    assert exhausted_list[0] == 1
    assert exhausted_list[1:3] == [2, 3]

    # Test with out-of-range index
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test with empty slice
    assert lazy_list[5:5] == []
    assert lazy_list[10:20] == []

    # Test with step in slice
    assert lazy_list[0:10:2] == [0, 2, 4, 6, 8]
    assert lazy_list[1:10:3] == [1, 4, 7]


# LLM-generated content at query #28
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[-1] == 9

    # Test slice access
    assert lazy_list[2:5] == [2, 3, 4]
    assert lazy_list[:3] == [0, 1, 2]
    assert lazy_list[3:] == [3, 4, 5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test with non-sequential access
    lazy_list = LazyList(range(10))
    assert lazy_list[5] == 5
    assert lazy_list[2] == 2
    assert lazy_list[8] == 8

    # Test with exhausted list
    lazy_list = LazyList(range(5))
    list(lazy_list)  # Exhaust the iterator
    assert lazy_list[0] == 0
    assert lazy_list[4] == 4
    assert lazy_list[2:4] == [2, 3]

    # Test with empty slice
    assert lazy_list[10:20] == []

    # Test with negative indices
    assert lazy_list[-2] == 3
    assert lazy_list[-4:-1] == [1, 2, 3]

    # Test with step in slice
    assert lazy_list[::-1] == [4, 3, 2, 1, 0]
    assert lazy_list[1:4:2] == [1, 3]


# LLM-generated content at query #29
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5])) == [3, 4, 5]

    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []

    # Test when no element satisfies the predicate
    assert list(drop_until(lambda x: x > 10, range(5))) == []

    # Test when the first element satisfies the predicate
    assert list(drop_until(lambda x: x == 1, [1, 2, 3])) == [1, 2, 3]

    # Test with a custom object
    class Custom:
        def __init__(self, val):
            self.val = val

    assert list(drop_until(lambda x: x.val > 2, [Custom(1), Custom(2), Custom(3), Custom(4)])) == [Custom(3), Custom(4)]

    # Test with a generator
    def gen():
        yield 1
        yield 2
        yield 3
        yield 4

    assert list(drop_until(lambda x: x == 3, gen())) == [3, 4]


# LLM-generated content at query #30
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:10] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test out of bounds access
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test empty slice
    assert lazy_list[5:5] == []

    # Test with non-sequential iterable
    lazy_list = LazyList([1, 4, 9, 16, 25])
    assert lazy_list[0] == 1
    assert lazy_list[2] == 9
    assert lazy_list[1:4] == [4, 9, 16]

    # Test with exhausted iterable
    lazy_list = LazyList(range(5))
    _ = list(lazy_list)  # Exhaust the iterable
    assert lazy_list[0] == 0
    assert lazy_list[4] == 4
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_split_by():
    # Test with criterion
    assert list(split_by(range(10), criterion=lambda x: x % 3 == 0)) == [[1, 2], [4, 5], [7, 8]]
    assert list(split_by(range(10), empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[1, 2], [], [4, 5], [], [7, 8], []]

    # Test with separator
    assert list(split_by(" Split by: ", separator=' ')) == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':']]
    assert list(split_by(" Split by: ", empty_segments=True, separator=' ')) == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

    # Test error when neither criterion nor separator is provided
    try:
        list(split_by(range(10)))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test error when both criterion and separator are provided
    try:
        list(split_by(range(10), criterion=lambda x: x % 3 == 0, separator=0))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with empty iterable
    assert list(split_by([], criterion=lambda x: x % 3 == 0)) == []
    assert list(split_by([], empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[]]

    # Test with all elements matching criterion
    assert list(split_by([0, 3, 6, 9], criterion=lambda x: x % 3 == 0)) == []
    assert list(split_by([0, 3, 6, 9], empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[], [], [], [], []]

    # Test with no elements matching criterion
    assert list(split_by([1, 2, 4, 5], criterion=lambda x: x % 3 == 0)) == [[1, 2, 4, 5]]
    assert list(split_by([1, 2, 4, 5], empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[1, 2, 4, 5]]


# LLM-generated content at query #2
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test negative index access
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test slice access
    assert lazy_list[2:5] == [2, 3, 4]
    assert lazy_list[:3] == [0, 1, 2]
    assert lazy_list[5:] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = lazy_list[10]
    with pytest.raises(IndexError):
        _ = lazy_list[-11]

    # Test with non-sequential iterable
    lazy_list = LazyList(x for x in range(10) if x % 2 == 0)
    assert lazy_list[0] == 0
    assert lazy_list[1] == 2
    assert lazy_list[2] == 4
    assert lazy_list[:2] == [0, 2]

    # Test exhaustion
    lazy_list = LazyList(range(5))
    _ = lazy_list[4]  # Exhaust the iterator
    assert lazy_list.list == [0, 1, 2, 3, 4]
    assert lazy_list.exhausted is True
    assert lazy_list[0] == 0  # Should still work after exhaustion


# LLM-generated content at query #3
#--------------------------

```python
def test_drop():
    # Test dropping zero elements
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]

    # Test dropping all elements
    assert list(drop(3, [1, 2, 3])) == []

    # Test dropping some elements
    assert list(drop(2, [1, 2, 3, 4, 5])) == [3, 4, 5]

    # Test dropping more elements than available
    assert list(drop(10, [1, 2, 3])) == []

    # Test with empty iterable
    assert list(drop(5, [])) == []

    # Test with negative n (should raise ValueError)
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with generator
    gen = (x for x in [1, 2, 3, 4, 5])
    assert list(drop(2, gen)) == [3, 4, 5]

    # Test with string
    assert list(drop(3, "hello")) == ['l', 'o']


# LLM-generated content at query #4
#--------------------------

```python
def test_split_by():
    # Test with criterion function
    assert list(split_by(range(10), criterion=lambda x: x % 3 == 0)) == [[1, 2], [4, 5], [7, 8]]
    assert list(split_by(range(10), empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[1, 2], [], [4, 5], [], [7, 8], []]

    # Test with separator
    assert list(split_by(" Split by: ", separator=' ')) == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':']]
    assert list(split_by(" Split by: ", empty_segments=True, separator=' ')) == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

    # Test with empty iterable
    assert list(split_by([], criterion=lambda x: x > 0)) == []
    assert list(split_by([], separator=0)) == []

    # Test with all elements matching criterion
    assert list(split_by([1, 2, 3], criterion=lambda x: x > 0)) == [[], [], []]
    assert list(split_by([1, 2, 3], empty_segments=False, criterion=lambda x: x > 0)) == []

    # Test with no elements matching criterion
    assert list(split_by([1, 2, 3], criterion=lambda x: x > 10)) == [[1, 2, 3]]

    # Test with error cases
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x > 0, separator=0))
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    try:
        list(split_by([1, 2, 3]))
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_Range___iter__():
    # Test with single argument (stop)
    r1 = Range(5)
    assert list(r1) == [0, 1, 2, 3, 4]

    # Test with start and stop
    r2 = Range(1, 5)
    assert list(r2) == [1, 2, 3, 4]

    # Test with start, stop, and step
    r3 = Range(1, 10, 2)
    assert list(r3) == [1, 3, 5, 7, 9]

    # Test with negative step (should be empty since val starts at l and step is positive)
    r4 = Range(5, 1, -1)
    assert list(r4) == []

    # Test with empty range
    r5 = Range(0)
    assert list(r5) == []

    # Test with large range (ensure it doesn't hang)
    r6 = Range(1000000)
    assert len(list(r6)) == 1000000
    assert list(r6) == []  # Ensure it's exhausted after first iteration


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Range___iter__():
    # Test with single argument
    r = Range(5)
    assert list(r) == [0, 1, 2, 3, 4]

    # Test with start and stop
    r = Range(1, 5)
    assert list(r) == [1, 2, 3, 4]

    # Test with start, stop, and step
    r = Range(1, 10, 2)
    assert list(r) == [1, 3, 5, 7, 9]

    # Test with negative step (should not work as per implementation)
    r = Range(5, 0, -1)
    assert list(r) == []

    # Test empty range
    r = Range(0)
    assert list(r) == []

    # Test with step larger than range
    r = Range(0, 10, 20)
    assert list(r) == [0]

    # Test that iterator can be consumed multiple times
    r = Range(3)
    assert list(r) == [0, 1, 2]
    assert list(r) == [0, 1, 2]


# LLM-generated content at query #2
#--------------------------

```python
def test_split_by():
    # Test with criterion
    assert list(split_by(range(10), criterion=lambda x: x % 3 == 0)) == [[1, 2], [4, 5], [7, 8]]
    assert list(split_by(range(10), empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[1, 2], [], [4, 5], [], [7, 8], []]

    # Test with separator
    assert list(split_by(" Split by: ", separator=' ')) == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':']]
    assert list(split_by(" Split by: ", empty_segments=True, separator=' ')) == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

    # Test with empty iterable
    assert list(split_by([], criterion=lambda x: x > 0)) == []
    assert list(split_by([], empty_segments=True, separator=0)) == [[]]

    # Test with all elements satisfying criterion
    assert list(split_by([1, 2, 3], criterion=lambda x: x > 0)) == []
    assert list(split_by([1, 2, 3], empty_segments=True, criterion=lambda x: x > 0)) == [[], [], [], []]

    # Test with no elements satisfying criterion
    assert list(split_by([1, 2, 3], criterion=lambda x: x > 10)) == [[1, 2, 3]]
    assert list(split_by([1, 2, 3], empty_segments=True, separator=0)) == [[1, 2, 3]]

    # Test with ValueError for invalid arguments
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x > 0, separator=0))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        list(split_by([1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_Range___next__():
    # Test basic iteration
    r = Range(5)
    assert next(r) == 0
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    # Test with start and stop
    r = Range(2, 5)
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    # Test with step
    r = Range(0, 10, 2)
    assert next(r) == 0
    assert next(r) == 2
    assert next(r) == 4
    assert next(r) == 6
    assert next(r) == 8
    with pytest.raises(StopIteration):
        next(r)

    # Test empty range
    r = Range(0)
    with pytest.raises(StopIteration):
        next(r)

    # Test negative step (though Range doesn't support it, this is just to verify behavior)
    r = Range(5, 0, -1)
    assert next(r) == 5
    assert next(r) == 4
    assert next(r) == 3
    assert next(r) == 2
    assert next(r) == 1
    with pytest.raises(StopIteration):
        next(r)


# LLM-generated content at query #4
#--------------------------

```python
def test_take():
    # Test basic functionality
    assert list(take(3, [1, 2, 3, 4, 5])) == [1, 2, 3]
    assert list(take(0, [1, 2, 3])) == []
    assert list(take(5, [1, 2, 3])) == [1, 2, 3]

    # Test with generator
    assert list(take(2, (x for x in range(10)))) == [0, 1]

    # Test with negative n
    try:
        list(take(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with empty iterable
    assert list(take(5, [])) == []

    # Test with string
    assert list(take(3, "hello")) == ['h', 'e', 'l']


# LLM-generated content at query #5
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(5, 5)) == 0
    assert len(Range(1, 10, 100)) == 1


# LLM-generated content at query #6
#--------------------------

```python
def test_Range___iter__():
    # Test with single argument
    r = Range(5)
    assert list(r) == [0, 1, 2, 3, 4]

    # Test with start and stop
    r = Range(1, 5)
    assert list(r) == [1, 2, 3, 4]

    # Test with start, stop, and step
    r = Range(1, 10, 2)
    assert list(r) == [1, 3, 5, 7, 9]

    # Test with negative step (should not work as __next__ doesn't handle it)
    r = Range(5, 0, -1)
    assert list(r) == []

    # Test with empty range
    r = Range(0)
    assert list(r) == []

    # Test with step larger than range
    r = Range(1, 5, 10)
    assert list(r) == [1]

    # Test that iterator is independent of original Range
    r = Range(3)
    it1 = iter(r)
    it2 = iter(r)
    assert next(it1) == 0
    assert next(it2) == 0
    assert next(it1) == 1
    assert next(it2) == 1


# LLM-generated content at query #7
#--------------------------

```python
def test_chunk():
    # Test basic chunking
    assert list(chunk(3, range(10))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    # Test chunk size larger than iterable
    assert list(chunk(20, range(10))) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    # Test chunk size of 1
    assert list(chunk(1, range(5))) == [[0], [1], [2], [3], [4]]

    # Test empty iterable
    assert list(chunk(3, [])) == []

    # Test chunk size equal to iterable length
    assert list(chunk(5, range(5))) == [[0, 1, 2, 3, 4]]

    # Test with non-integer iterable
    assert list(chunk(2, ['a', 'b', 'c', 'd', 'e'])) == [['a', 'b'], ['c', 'd'], ['e']]

    # Test with generator
    assert list(chunk(3, (x for x in range(10)))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    # Test with invalid chunk size (should raise ValueError)
    try:
        list(chunk(0, range(10)))
        assert False, "Expected ValueError for n <= 0"
    except ValueError:
        pass

    try:
        list(chunk(-1, range(10)))
        assert False, "Expected ValueError for n <= 0"
    except ValueError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_Range___getitem__():
    # Test single index access
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[-1] == 9

    # Test slice access
    assert r[1:4] == [1, 2, 3]
    assert r[:3] == [0, 1, 2]
    assert r[3:] == [3, 4, 5, 6, 7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1:5:2] == [1, 3]

    # Test with start, stop, step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[-1] == 9
    assert r[1:3] == [3, 5]
    assert r[:2] == [1, 3]
    assert r[2:] == [5, 7, 9]

    # Test empty range
    r = Range(0)
    with pytest.raises(IndexError):
        _ = r[0]
    assert r[:] == []

    # Test negative indices
    r = Range(5)
    assert r[-1] == 4
    assert r[-2] == 3
    assert r[-5] == 0
    with pytest.raises(IndexError):
        _ = r[-6]

    # Test out of bounds
    r = Range(3)
    with pytest.raises(IndexError):
        _ = r[3]
    with pytest.raises(IndexError):
        _ = r[100]


# LLM-generated content at query #9
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(0, 0, 1)) == 0
    assert len(Range(5, 5)) == 0
    assert len(Range(5, 5, 1)) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_Range___next__():
    # Test normal iteration
    r = Range(1, 5)
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    # Test with step
    r = Range(0, 10, 2)
    assert next(r) == 0
    assert next(r) == 2
    assert next(r) == 4
    assert next(r) == 6
    assert next(r) == 8
    with pytest.raises(StopIteration):
        next(r)

    # Test single argument
    r = Range(3)
    assert next(r) == 0
    assert next(r) == 1
    assert next(r) == 2
    with pytest.raises(StopIteration):
        next(r)

    # Test negative step (should raise StopIteration immediately)
    r = Range(5, 1, -1)
    with pytest.raises(StopIteration):
        next(r)


# LLM-generated content at query #11
#--------------------------

```python
def test_drop():
    # Test dropping elements from a list
    assert list(drop(3, [1, 2, 3, 4, 5])) == [4, 5]
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    assert list(drop(5, [1, 2, 3])) == []

    # Test dropping elements from a generator
    assert list(drop(2, (x for x in range(10)))) == [2, 3, 4, 5, 6, 7, 8, 9]

    # Test dropping all elements
    assert list(drop(10, range(5))) == []

    # Test dropping with negative n (should raise ValueError)
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test dropping from an empty iterable
    assert list(drop(5, [])) == []


# LLM-generated content at query #12
#--------------------------

```python
def test_drop_until():
    # Test dropping until a condition is met
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    assert list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5])) == [3, 4, 5]
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7])) == [6, 7]

    # Test when no element satisfies the condition
    assert list(drop_until(lambda x: x > 10, range(5))) == []

    # Test when the first element satisfies the condition
    assert list(drop_until(lambda x: x == 1, [1, 2, 3])) == [1, 2, 3]

    # Test with an empty iterable
    assert list(drop_until(lambda x: x > 0, [])) == []

    # Test with a custom object
    class Custom:
        def __init__(self, value):
            self.value = value

    objects = [Custom(1), Custom(2), Custom(3)]
    assert list(drop_until(lambda x: x.value == 2, objects)) == [Custom(2), Custom(3)]


# LLM-generated content at query #13
#--------------------------

```python
def test_Range___len__():
    # Test single argument
    assert len(Range(10)) == 10
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test two arguments
    assert len(Range(1, 11)) == 10
    assert len(Range(5, 5)) == 0
    assert len(Range(-3, 3)) == 6

    # Test three arguments (with positive step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 11, 3)) == 4
    assert len(Range(5, 5, 1)) == 0

    # Test three arguments (with negative step)
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(5, -5, -2)) == 5
    assert len(Range(3, 3, -1)) == 0

    # Test edge cases
    assert len(Range(0, 0, 1)) == 0
    assert len(Range(10, 10, -1)) == 0
    assert len(Range(1, 100, 100)) == 1


# LLM-generated content at query #14
#--------------------------

```python
def test_Range___iter__():
    # Test with single argument (stop)
    r1 = Range(5)
    assert list(r1) == [0, 1, 2, 3, 4]

    # Test with start and stop
    r2 = Range(2, 5)
    assert list(r2) == [2, 3, 4]

    # Test with start, stop, and step
    r3 = Range(1, 10, 2)
    assert list(r3) == [1, 3, 5, 7, 9]

    # Test with negative step
    r4 = Range(5, 0, -1)
    assert list(r4) == [5, 4, 3, 2, 1]

    # Test with empty range
    r5 = Range(0)
    assert list(r5) == []

    # Test with step larger than range
    r6 = Range(0, 10, 20)
    assert list(r6) == [0]

    # Test that iterator can be called multiple times
    r7 = Range(3)
    assert list(r7) == [0, 1, 2]
    assert list(r7) == [0, 1, 2]


# LLM-generated content at query #15
#--------------------------

```python
def test_Range___iter__():
    # Test basic iteration
    r = Range(5)
    assert list(r) == [0, 1, 2, 3, 4]

    # Test iteration with start and stop
    r = Range(1, 5)
    assert list(r) == [1, 2, 3, 4]

    # Test iteration with start, stop, and step
    r = Range(1, 10, 2)
    assert list(r) == [1, 3, 5, 7, 9]

    # Test iteration with negative step (should not work as __iter__ uses Range constructor)
    r = Range(5, 0, -1)
    assert list(r) == []  # Since __iter__ creates a new Range with default step=1

    # Test empty range
    r = Range(0)
    assert list(r) == []

    # Test single element range
    r = Range(1)
    assert list(r) == [0]

    # Test large range (to ensure it doesn't hang)
    r = Range(1000)
    assert len(list(r)) == 1000


# LLM-generated content at query #16
#--------------------------

```python
def test_Range___next__():
    r = Range(1, 5)
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    r = Range(1, 5, 2)
    assert next(r) == 1
    assert next(r) == 3
    with pytest.raises(StopIteration):
        next(r)

    r = Range(5)
    assert next(r) == 0
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)


# LLM-generated content at query #17
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(5)) == 5
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0  # Negative stop should result in empty range

    # Test with two arguments (start, stop)
    assert len(Range(1, 5)) == 4
    assert len(Range(0, 0)) == 0
    assert len(Range(-3, 3)) == 6
    assert len(Range(5, 1)) == 0  # Start > stop should result in empty range

    # Test with three arguments (start, stop, step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(0, 10, 1)) == 10
    assert len(Range(0, 10, 11)) == 1
    assert len(Range(0, 10, 100)) == 1
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(10, 0, 1)) == 0  # Positive step with start > stop
    assert len(Range(0, 10, -1)) == 0  # Negative step with start < stop


# LLM-generated content at query #18
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5

    # Test slice with positive indices
    assert r[1:5] == [1, 2, 3, 4]
    assert r[:5] == [0, 1, 2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]

    # Test slice with negative indices
    assert r[-5:-1] == [5, 6, 7, 8]
    assert r[-5:] == [5, 6, 7, 8, 9]
    assert r[:-1] == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Test with step
    r_step = Range(0, 10, 2)
    assert r_step[0] == 0
    assert r_step[2] == 4
    assert r_step[-1] == 8
    assert r_step[1:4] == [2, 4, 6]

    # Test out of bounds (should raise IndexError)
    try:
        _ = r[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        _ = r[-11]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test empty slice
    assert r[5:5] == []
    assert r[10:20] == []


# LLM-generated content at query #19
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(5, 5)) == 0
    assert len(Range(5, 5, 1)) == 0
    assert len(Range(5, 5, -1)) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5
    assert r[-10] == 0

    # Test slice with positive indices
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:7] == [2, 3, 4, 5, 6]
    assert r[5:] == [5, 6, 7, 8, 9]

    # Test slice with negative indices
    assert r[-5:-1] == [5, 6, 7, 8]
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]

    # Test slice with step
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Test Range with start and stop
    r = Range(5, 10)
    assert r[0] == 5
    assert r[2] == 7
    assert r[-1] == 9
    assert r[1:4] == [6, 7, 8]

    # Test Range with start, stop, and step
    r = Range(0, 10, 2)
    assert r[0] == 0
    assert r[2] == 4
    assert r[-1] == 8
    assert r[1:4] == [2, 4, 6]

    # Test out of bounds index
    with pytest.raises(IndexError):
        r[10]
    with pytest.raises(IndexError):
        r[-11]


# LLM-generated content at query #21
#--------------------------

```python
def test_Range___getitem__():
    # Test single positive index
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9

    # Test single negative index
    assert r[-1] == 9
    assert r[-5] == 5
    assert r[-10] == 0

    # Test slice with positive indices
    assert r[1:5] == [1, 2, 3, 4]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[2:8:3] == [2, 5]

    # Test slice with negative indices
    assert r[-5:-1] == [5, 6, 7, 8]
    assert r[-1:-5:-1] == [9, 8, 7, 6]

    # Test Range with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[5] == 10
    assert r[-1] == 14
    assert r[2:7] == [7, 8, 9, 10, 11]

    # Test Range with start, stop, and step
    r = Range(0, 20, 2)
    assert r[0] == 0
    assert r[5] == 10
    assert r[-1] == 18
    assert r[3:8] == [6, 8, 10, 12, 14]

    # Test out of bounds index
    with pytest.raises(IndexError):
        r[20]
    with pytest.raises(IndexError):
        r[-20]

    # Test empty slice
    assert r[10:5] == []


# LLM-generated content at query #22
#--------------------------

```python
def test_LazyList___getitem__():
    # Test single index access
    ll = LazyList(range(10))
    assert ll[0] == 0
    assert ll[5] == 5
    assert ll[9] == 9

    # Test negative index access
    assert ll[-1] == 9
    assert ll[-5] == 5

    # Test slice access
    assert ll[2:5] == [2, 3, 4]
    assert ll[:3] == [0, 1, 2]
    assert ll[5:] == [5, 6, 7, 8, 9]
    assert ll[::2] == [0, 2, 4, 6, 8]
    assert ll[1::2] == [1, 3, 5, 7, 9]

    # Test out of bounds
    try:
        _ = ll[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with non-sequential access
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[4] == 5
    assert ll[0] == 1
    assert ll[2] == 3

    # Test with exhausted iterator
    ll = LazyList([1, 2, 3])
    list(ll)  # Exhaust the iterator
    assert ll[0] == 1
    assert ll[1] == 2
    assert ll[2] == 3


# LLM-generated content at query #23
#--------------------------

```python
def test_Range___len__():
    assert len(Range(10)) == 10
    assert len(Range(1, 10)) == 9
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(10, 1, -1)) == 9
    assert len(Range(10, 1, -2)) == 5
    assert len(Range(0)) == 0
    assert len(Range(0, 0)) == 0
    assert len(Range(0, 0, 1)) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_split_by():
    # Test with criterion
    assert list(split_by(range(10), criterion=lambda x: x % 3 == 0)) == [[1, 2], [4, 5], [7, 8]]
    assert list(split_by(range(10), empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[1, 2], [], [4, 5], [], [7, 8], []]

    # Test with separator
    assert list(split_by(" Split by: ", separator=' ')) == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':']]
    assert list(split_by(" Split by: ", empty_segments=True, separator=' ')) == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

    # Test with empty iterable
    assert list(split_by([], criterion=lambda x: x % 3 == 0)) == []
    assert list(split_by([], empty_segments=True, separator=' ')) == []

    # Test with all elements matching criterion
    assert list(split_by([3, 6, 9], criterion=lambda x: x % 3 == 0)) == []
    assert list(split_by([3, 6, 9], empty_segments=True, criterion=lambda x: x % 3 == 0)) == [[], [], [], []]

    # Test with no elements matching criterion
    assert list(split_by([1, 2, 4, 5], criterion=lambda x: x % 3 == 0)) == [[1, 2, 4, 5]]
    assert list(split_by([1, 2, 4, 5], empty_segments=True, separator=' ')) == [[1, 2, 4, 5]]

    # Test with ValueError for invalid arguments
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x % 3 == 0, separator=' '))
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        list(split_by([1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_Range___next__():
    # Test basic iteration
    r = Range(5)
    assert next(r) == 0
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass

    # Test with start and stop
    r = Range(2, 5)
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass

    # Test with step
    r = Range(0, 10, 2)
    assert next(r) == 0
    assert next(r) == 2
    assert next(r) == 4
    assert next(r) == 6
    assert next(r) == 8
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass

    # Test negative step (should not work as per implementation)
    r = Range(5, 0, -1)
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(10)) == 10
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test with two arguments (start, stop)
    assert len(Range(1, 10)) == 9
    assert len(Range(0, 0)) == 0
    assert len(Range(-5, 5)) == 10
    assert len(Range(5, 1)) == 0  # Invalid range (start > stop)

    # Test with three arguments (start, stop, step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(0, 10, -1)) == 0  # Invalid step (negative with start < stop)
    assert len(Range(0, 10, 0)) == 0  # Zero step (should not happen in practice)


# LLM-generated content at query #27
#--------------------------

```python
def test_Range___len__():
    # Test with single argument (stop)
    assert len(Range(10)) == 10
    assert len(Range(0)) == 0
    assert len(Range(-5)) == 0

    # Test with two arguments (start, stop)
    assert len(Range(1, 10)) == 9
    assert len(Range(0, 0)) == 0
    assert len(Range(-5, 5)) == 10
    assert len(Range(10, 1)) == 0

    # Test with three arguments (start, stop, step)
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 5
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(0, 10, -1)) == 0
    assert len(Range(0, 10, 100)) == 1


# LLM-generated content at query #28
#--------------------------

```python
def test_Range___next__():
    # Test normal iteration
    r = Range(5)
    assert next(r) == 0
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    # Test with start and stop
    r = Range(1, 5)
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4
    with pytest.raises(StopIteration):
        next(r)

    # Test with step
    r = Range(0, 10, 2)
    assert next(r) == 0
    assert next(r) == 2
    assert next(r) == 4
    assert next(r) == 6
    assert next(r) == 8
    with pytest.raises(StopIteration):
        next(r)

    # Test with negative step (should not iterate)
    r = Range(5, 0, -1)
    with pytest.raises(StopIteration):
        next(r)


# LLM-generated content at query #29
#--------------------------

```python
def test_MapList___getitem__():
    # Test with integer index
    lst = [1, 2, 3, 4, 5]
    func = lambda x: x * 2
    map_list = MapList(func, lst)
    assert map_list[0] == 2
    assert map_list[2] == 6
    assert map_list[-1] == 10

    # Test with slice
    assert map_list[1:4] == [4, 6, 8]
    assert map_list[:3] == [2, 4, 6]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[::2] == [2, 6, 10]

    # Test with empty slice
    assert map_list[5:10] == []
    assert map_list[10:20] == []

    # Test with negative indices in slice
    assert map_list[-3:-1] == [6, 8]
    assert map_list[-5:-2] == [2, 4, 6, 8]

    # Test with step in slice
    assert map_list[0:5:2] == [2, 6, 10]
    assert map_list[1:5:2] == [4, 8]


# LLM-generated content at query #30
#--------------------------

```python
def test_LazyList___getitem__():
    # Test with integer index
    lazy_list = LazyList(range(10))
    assert lazy_list[0] == 0
    assert lazy_list[5] == 5
    assert lazy_list[9] == 9

    # Test with negative integer index
    assert lazy_list[-1] == 9
    assert lazy_list[-5] == 5

    # Test with slice
    assert lazy_list[0:5] == [0, 1, 2, 3, 4]
    assert lazy_list[5:10] == [5, 6, 7, 8, 9]
    assert lazy_list[::2] == [0, 2, 4, 6, 8]
    assert lazy_list[1::2] == [1, 3, 5, 7, 9]

    # Test with slice and negative indices
    assert lazy_list[-5:-1] == [5, 6, 7, 8]
    assert lazy_list[-1:-5:-1] == [9, 8, 7, 6]

    # Test with slice and step
    assert lazy_list[0:10:2] == [0, 2, 4, 6, 8]
    assert lazy_list[1:10:2] == [1, 3, 5, 7, 9]

    # Test with slice and negative step
    assert lazy_list[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert lazy_list[5::-1] == [5, 4, 3, 2, 1, 0]

    # Test with slice and negative start and stop
    assert lazy_list[-1:-5:-1] == [9, 8, 7, 6]

    # Test with slice and negative start and stop and step
    assert lazy_list[-1:-5:-2] == [9, 7]

    # Test with slice and negative start and stop and negative step
    assert lazy_list[-5:-1:-1] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:1] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:2] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:3] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:4] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:5] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:6] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:7] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:8] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:9] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:10] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:11] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:12] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:13] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:14] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:15] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:16] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:17] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:18] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:19] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:20] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:21] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:22] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:23] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:24] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:25] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:26] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:27] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:28] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:29] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:30] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:31] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:32] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:33] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:34] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:35] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:36] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:37] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:38] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:39] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:40] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:41] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:42] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:43] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:44] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:45] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:46] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:47] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:48] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:49] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:50] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:51] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:52] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:53] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:54] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:55] == []

    # Test with slice and negative start and stop and positive step
    assert lazy_list[-5:-1:56] ==


