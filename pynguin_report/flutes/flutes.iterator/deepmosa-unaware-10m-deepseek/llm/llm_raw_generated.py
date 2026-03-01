####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Range___len__():
    # Test basic length with single argument (stop)
    assert len(Range(5)) == 5
    assert len(Range(0)) == 0
    assert len(Range(10)) == 10
    
    # Test length with start and stop
    assert len(Range(2, 7)) == 5
    assert len(Range(5, 5)) == 0
    assert len(Range(5, 10)) == 5
    assert len(Range(-3, 2)) == 5
    
    # Test length with start, stop, and step
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 4
    assert len(Range(0, 10, 3)) == 4
    assert len(Range(0, 10, 5)) == 2
    assert len(Range(0, 10, 10)) == 1
    assert len(Range(0, 10, 11)) == 0
    
    # Test negative step
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(10, 0, -3)) == 4
    
    # Test edge cases with negative start/stop
    assert len(Range(-5, 0)) == 5
    assert len(Range(-10, -5)) == 5
    assert len(Range(-10, 0, 2)) == 5
    assert len(Range(-10, 0, 3)) == 4
    
    # Test that length matches iteration count
    r = Range(3, 15, 4)
    assert len(r) == sum(1 for _ in r)
    
    # Test that length is integer division
    assert len(Range(0, 7, 2)) == 3  # 0, 2, 4, 6
    assert len(Range(0, 6, 2)) == 3  # 0, 2, 4
    assert len(Range(0, 5, 2)) == 2  # 0, 2, 4


# LLM-generated content at query #2
#--------------------------

```python
def test_Range___iter__():
    # Test basic iteration with stop only
    assert list(Range(5)) == [0, 1, 2, 3, 4]
    
    # Test iteration with start and stop
    assert list(Range(2, 7)) == [2, 3, 4, 5, 6]
    
    # Test iteration with start, stop, and step
    assert list(Range(1, 10, 2)) == [1, 3, 5, 7, 9]
    
    # Test iteration with negative step
    assert list(Range(5, 0, -1)) == [5, 4, 3, 2, 1]
    
    # Test iteration with empty range
    assert list(Range(0)) == []
    assert list(Range(5, 5)) == []
    assert list(Range(5, 0)) == []
    
    # Test iteration with large step
    assert list(Range(0, 10, 3)) == [0, 3, 6, 9]
    
    # Test that multiple iterations work independently
    r = Range(3)
    iter1 = iter(r)
    iter2 = iter(r)
    assert list(iter1) == [0, 1, 2]
    assert list(iter2) == [0, 1, 2]
    
    # Test iteration with negative start
    assert list(Range(-3, 0)) == [-3, -2, -1]
    
    # Test iteration with mixed signs
    assert list(Range(-2, 3)) == [-2, -1, 0, 1, 2]
    
    # Test that iteration doesn't affect indexing
    r = Range(5)
    list(r)  # Consume the iterator
    assert r[0] == 0
    assert r[4] == 4


# LLM-generated content at query #3
#--------------------------

```python
def test_chunk():
    # Test basic chunking
    assert list(chunk(3, range(10))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    
    # Test chunk size larger than iterable
    assert list(chunk(5, [1, 2, 3])) == [[1, 2, 3]]
    
    # Test chunk size equal to iterable length
    assert list(chunk(3, [1, 2, 3])) == [[1, 2, 3]]
    
    # Test chunk size of 1
    assert list(chunk(1, [1, 2, 3])) == [[1], [2], [3]]
    
    # Test empty iterable
    assert list(chunk(3, [])) == []
    
    # Test with iterator
    assert list(chunk(2, iter([1, 2, 3, 4]))) == [[1, 2], [3, 4]]
    
    # Test with string
    assert list(chunk(2, "abcde")) == [['a', 'b'], ['c', 'd'], ['e']]
    
    # Test with generator
    assert list(chunk(2, (x for x in range(5)))) == [[0, 1], [2, 3], [4]]
    
    # Test invalid chunk size
    try:
        list(chunk(0, [1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "`n` should be positive"
    
    try:
        list(chunk(-1, [1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "`n` should be positive"


# LLM-generated content at query #4
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied immediately
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda c: c == 'l', "hello world")) == ['l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    
    # Test with list
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7, 8])) == [6, 7, 8]
    
    # Test with generator
    gen = (x for x in range(10))
    assert list(drop_until(lambda x: x > 7, gen)) == [8, 9]
    
    # Test with single element that satisfies predicate
    assert list(drop_until(lambda x: x == 5, [5])) == [5]
    
    # Test with single element that doesn't satisfy predicate
    assert list(drop_until(lambda x: x == 5, [3])) == []
    
    # Test with predicate that checks type
    assert list(drop_until(lambda x: isinstance(x, str), [1, 2, "hello", "world"])) == ["hello", "world"]


# LLM-generated content at query #5
#--------------------------

```python
def test_Range___iter__():
    # Test basic iteration with stop only
    assert list(Range(5)) == [0, 1, 2, 3, 4]
    
    # Test iteration with start and stop
    assert list(Range(2, 7)) == [2, 3, 4, 5, 6]
    
    # Test iteration with start, stop, and step
    assert list(Range(1, 10, 2)) == [1, 3, 5, 7, 9]
    
    # Test iteration with negative step
    assert list(Range(5, 0, -1)) == [5, 4, 3, 2, 1]
    
    # Test iteration with empty range
    assert list(Range(0)) == []
    assert list(Range(5, 5)) == []
    assert list(Range(5, 0)) == []
    
    # Test iteration with step that skips all elements
    assert list(Range(0, 10, 20)) == [0]
    assert list(Range(1, 10, 20)) == [1]
    
    # Test that multiple iterations work independently
    r = Range(3)
    it1 = iter(r)
    it2 = iter(r)
    assert list(it1) == [0, 1, 2]
    assert list(it2) == [0, 1, 2]
    
    # Test iteration with negative start
    assert list(Range(-3, 0)) == [-3, -2, -1]
    
    # Test iteration with mixed signs
    assert list(Range(-2, 3)) == [-2, -1, 0, 1, 2]
    
    # Test that iteration doesn't affect indexing
    r = Range(5)
    _ = list(iter(r))
    assert r[0] == 0
    assert r[4] == 4


# LLM-generated content at query #6
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied immediately
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda c: c == 'l', "hello world")) == ['l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    
    # Test with list
    assert list(drop_until(lambda x: x % 2 == 1, [2, 4, 6, 7, 8, 9])) == [7, 8, 9]
    
    # Test with generator
    gen = (x for x in range(10))
    assert list(drop_until(lambda x: x > 7, gen)) == [8, 9]
    
    # Test with single element that satisfies predicate
    assert list(drop_until(lambda x: x == 5, [5])) == [5]
    
    # Test with single element that doesn't satisfy predicate
    assert list(drop_until(lambda x: x == 5, [3])) == []
    
    # Test with all elements satisfying predicate
    assert list(drop_until(lambda x: True, [1, 2, 3])) == [1, 2, 3]
    
    # Test with no elements satisfying predicate
    assert list(drop_until(lambda x: False, [1, 2, 3])) == []


# LLM-generated content at query #7
#--------------------------

```python
def test_drop():
    # Test dropping zero elements
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    
    # Test dropping from beginning
    assert list(drop(2, [1, 2, 3, 4, 5])) == [3, 4, 5]
    
    # Test dropping more elements than available
    assert list(drop(10, [1, 2, 3])) == []
    
    # Test dropping all elements
    assert list(drop(3, [1, 2, 3])) == []
    
    # Test with generator
    assert list(drop(2, (x for x in range(5)))) == [2, 3, 4]
    
    # Test with empty iterable
    assert list(drop(5, [])) == []
    
    # Test with negative n (should raise ValueError)
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with string
    assert list(drop(2, "hello")) == ['l', 'l', 'o']
    
    # Test with range
    assert list(drop(5, range(10))) == [5, 6, 7, 8, 9]
    
    # Test that it returns iterator, not list
    result = drop(1, [1, 2, 3])
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')
    
    # Test dropping from single element list
    assert list(drop(1, [42])) == []
    
    # Test dropping zero from empty list
    assert list(drop(0, [])) == []


# LLM-generated content at query #8
#--------------------------

```python
def test_drop():
    # Test basic functionality
    assert list(drop(2, [1, 2, 3, 4, 5])) == [3, 4, 5]
    
    # Test dropping 0 elements
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]
    
    # Test dropping more elements than available
    assert list(drop(10, [1, 2, 3])) == []
    
    # Test with empty iterable
    assert list(drop(3, [])) == []
    
    # Test with generator
    assert list(drop(2, (x for x in range(5)))) == [2, 3, 4]
    
    # Test with string
    assert list(drop(2, "hello")) == ['l', 'l', 'o']
    
    # Test with range
    assert list(drop(5, range(10))) == [5, 6, 7, 8, 9]
    
    # Test negative n raises ValueError
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test that it's lazy (doesn't consume entire iterable immediately)
    def counting_gen():
        for i in range(10):
            yield i
            if i > 5:
                raise AssertionError("Should not have been called")
    
    result = list(drop(6, counting_gen()))
    assert result == [6, 7, 8, 9]
    
    # Test with large n
    assert list(drop(1000, range(1005))) == [1000, 1001, 1002, 1003, 1004]
    
    # Test dropping all elements
    assert list(drop(5, [1, 2, 3, 4, 5])) == []
    
    # Test with iterator (should work after partial consumption)
    it = iter([1, 2, 3, 4, 5])
    next(it)  # Consume first element
    assert list(drop(1, it)) == [3, 4, 5]


# LLM-generated content at query #9
#--------------------------

```python
def test_split_by():
    # Test with criterion
    assert list(split_by(range(10), criterion=lambda x: x % 3 == 0)) == [
        [1, 2],
        [4, 5],
        [7, 8],
    ]
    
    # Test with separator
    assert list(split_by("hello world", separator=' ')) == [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']]
    
    # Test with empty_segments=True and separator
    assert list(split_by(" Split by: ", empty_segments=True, separator=' ')) == [
        [],
        ['S', 'p', 'l', 'i', 't'],
        ['b', 'y', ':'],
        [],
    ]
    
    # Test with empty_segments=True and criterion
    assert list(split_by([1, 2, 3, 4], empty_segments=True, criterion=lambda x: x % 2 == 0)) == [
        [1],
        [3],
    ]
    
    # Test empty iterable
    assert list(split_by([], criterion=lambda x: x)) == []
    
    # Test empty iterable with empty_segments=True
    assert list(split_by([], empty_segments=True, criterion=lambda x: x)) == []
    
    # Test no separators
    assert list(split_by([1, 2, 3], criterion=lambda x: x > 10)) == [[1, 2, 3]]
    
    # Test all separators with empty_segments=False
    assert list(split_by([0, 0, 0], criterion=lambda x: x == 0)) == []
    
    # Test all separators with empty_segments=True
    assert list(split_by([0, 0, 0], empty_segments=True, criterion=lambda x: x == 0)) == [[], [], [], []]
    
    # Test single element separator
    assert list(split_by([5], criterion=lambda x: x == 5)) == []
    
    # Test single element non-separator
    assert list(split_by([5], criterion=lambda x: x == 0)) == [[5]]
    
    # Test with string separator
    assert list(split_by("a.b.c", separator='.')) == [['a'], ['b'], ['c']]
    
    # Test error when both criterion and separator are None
    try:
        list(split_by([1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test error when both criterion and separator are specified
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x, separator=1))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_Range___len__():
    # Test basic length with single argument
    assert len(Range(5)) == 5
    assert len(Range(0)) == 0
    assert len(Range(1)) == 1
    
    # Test length with start and stop
    assert len(Range(2, 7)) == 5
    assert len(Range(0, 10)) == 10
    assert len(Range(5, 5)) == 0
    assert len(Range(5, 6)) == 1
    
    # Test length with step
    assert len(Range(0, 10, 2)) == 5
    assert len(Range(1, 10, 2)) == 4
    assert len(Range(0, 9, 3)) == 3
    assert len(Range(0, 10, 3)) == 4
    
    # Test negative step
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(10, 5, -1)) == 5
    assert len(Range(10, 5, -2)) == 3
    
    # Test edge cases with step
    assert len(Range(0, 10, 20)) == 1
    assert len(Range(0, 0, 5)) == 0
    assert len(Range(5, 5, 2)) == 0
    assert len(Range(5, 6, 2)) == 1
    
    # Test that length matches iteration count
    r = Range(3, 15, 4)
    assert len(r) == 3
    assert list(r) == [3, 7, 11]
    
    # Test with negative start
    assert len(Range(-5, 5)) == 10
    assert len(Range(-10, 0, 2)) == 5
    
    # Test with all negative
    assert len(Range(-10, -5)) == 5
    assert len(Range(-10, -5, 2)) == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test indexing with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[2] == 5
    assert r[3] == 7
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slicing with positive step
    r = Range(10)
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Test slicing with step parameter
    assert r[0:10:2] == [0, 2, 4, 6, 8]
    assert r[1:10:3] == [1, 4, 7]
    
    # Test negative slicing
    assert r[-5:] == [5, 6, 7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-7:-2] == [3, 4, 5, 6, 7]
    
    # Test slicing with start, stop, step
    r = Range(5, 20, 3)
    assert r[:] == [5, 8, 11, 14, 17]
    assert r[1:4] == [8, 11, 14]
    assert r[::2] == [5, 11, 17]
    
    # Test edge cases
    r = Range(3)
    assert r[0:0] == []
    assert r[5:10] == []
    assert r[-10:-5] == []
    
    # Test with negative step (should work with slicing)
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[5] == 5
    assert r[9] == 1
    assert r[:5] == [10, 9, 8, 7, 6]
    assert r[5:] == [5, 4, 3, 2, 1]
    
    # Test with larger negative step
    r = Range(20, 0, -3)
    assert r[0] == 20
    assert r[1] == 17
    assert r[2] == 14
    assert r[:3] == [20, 17, 14]
    
    # Test IndexError for out of bounds
    r = Range(5)
    try:
        r[5]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        r[-6]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with single element range
    r = Range(1, 2)
    assert r[0] == 1
    assert r[-1] == 1
    assert r[:] == [1]
    
    # Test with empty range
    r = Range(0)
    assert r[:] == []
    
    r = Range(5, 5)
    assert r[:] == []
    
    # Test complex slicing scenarios
    r = Range(100)
    assert r[10:20:2] == [10, 12, 14, 16, 18]
    assert r[-20:-10:3] == [80, 83, 86, 89]
    assert r[90::2] == [90, 92, 94, 96, 98]


# LLM-generated content at query #12
#--------------------------

```python
def test_drop_until():
    # Test basic functionality - drop until predicate is satisfied
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]
    
    # Test with empty iterable
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []
    
    # Test where predicate is never satisfied
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []
    
    # Test where predicate is satisfied immediately
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    result = list(drop_until(lambda c: c == 'b', 'abcdef'))
    assert result == ['b', 'c', 'd', 'e', 'f']
    
    # Test with list of strings
    result = list(drop_until(lambda s: len(s) > 2, ['a', 'ab', 'abc', 'abcd']))
    assert result == ['abc', 'abcd']
    
    # Test with generator expression
    result = list(drop_until(lambda x: x % 3 == 0, (x for x in range(10))))
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Test with predicate satisfied in middle
    result = list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5]))
    assert result == [3, 4, 5]
    
    # Test with single element list where predicate is satisfied
    result = list(drop_until(lambda x: x == 1, [1]))
    assert result == [1]
    
    # Test with single element list where predicate is not satisfied
    result = list(drop_until(lambda x: x == 2, [1]))
    assert result == []
    
    # Test with None values
    result = list(drop_until(lambda x: x is not None, [None, None, 1, 2, 3]))
    assert result == [1, 2, 3]


# LLM-generated content at query #13
#--------------------------

```python
def test_MapList___getitem__():
    # Test integer indexing
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * 2, lst)
    assert map_list[0] == 2
    assert map_list[2] == 6
    assert map_list[-1] == 10
    assert map_list[-2] == 8
    
    # Test slice indexing
    assert map_list[1:3] == [4, 6]
    assert map_list[:3] == [2, 4, 6]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[::2] == [2, 6, 10]
    assert map_list[::-1] == [10, 8, 6, 4, 2]
    
    # Test with different transformation
    map_list2 = MapList(lambda x: x**2, lst)
    assert map_list2[0] == 1
    assert map_list2[2] == 9
    assert map_list2[1:4] == [4, 9, 16]
    
    # Test with Range as input
    map_list3 = MapList(lambda x: x * 3, Range(5))
    assert map_list3[0] == 0
    assert map_list3[3] == 9
    assert map_list3[1:4] == [3, 6, 9]
    
    # Test with empty list
    map_list4 = MapList(lambda x: x + 1, [])
    assert map_list4[:] == []
    
    # Test with complex transformation
    lst2 = ["a", "b", "c"]
    map_list5 = MapList(lambda x: x * 2 + "_test", lst2)
    assert map_list5[0] == "aa_test"
    assert map_list5[1:] == ["bb_test", "cc_test"]
    
    # Test that original list is not modified
    original = [1, 2, 3]
    map_list6 = MapList(lambda x: x * 10, original)
    _ = map_list6[0]
    assert original == [1, 2, 3]


# LLM-generated content at query #14
#--------------------------

```python
def test_Range___getitem__():
    # Test single integer indexing
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-5] == 1
    
    # Test slice indexing
    r = Range(10)
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slice with negative indices
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slice with step
    assert r[1:9:3] == [1, 4, 7]
    assert r[::3] == [0, 3, 6, 9]
    assert r[2:10:2] == [2, 4, 6, 8]
    
    # Test slice with start, stop, step
    r = Range(5, 20, 3)
    assert r[:] == [5, 8, 11, 14, 17]
    assert r[1:4] == [8, 11, 14]
    assert r[::2] == [5, 11, 17]
    assert r[1::2] == [8, 14]
    
    # Test edge cases
    r = Range(3)
    assert r[0:0] == []
    assert r[5:10] == []
    assert r[2:1] == []
    
    # Test with negative step in constructor (should still work)
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[9] == 1
    assert r[:5] == [10, 9, 8, 7, 6]
    assert r[-1] == 1
    
    # Test single element range
    r = Range(1, 2)
    assert r[0] == 1
    assert r[:] == [1]
    assert r[-1] == 1
    
    # Test empty range
    r = Range(0)
    assert r[:] == []
    
    # Test indexing out of bounds should raise IndexError
    import pytest
    r = Range(5)
    with pytest.raises(IndexError):
        _ = r[5]
    with pytest.raises(IndexError):
        _ = r[-6]


# LLM-generated content at query #15
#--------------------------

```python
def test_Range___len__():
    # Test basic length calculation
    assert len(Range(5)) == 5
    assert len(Range(0, 5)) == 5
    assert len(Range(2, 7)) == 5
    assert len(Range(1, 10, 2)) == 4
    assert len(Range(0, 10, 3)) == 3
    
    # Test with negative step
    assert len(Range(10, 0, -1)) == 10
    assert len(Range(10, 0, -2)) == 5
    assert len(Range(5, -5, -2)) == 5
    
    # Test edge cases
    assert len(Range(0)) == 0
    assert len(Range(5, 5)) == 0
    assert len(Range(5, 5, 2)) == 0
    assert len(Range(0, 10, 11)) == 0
    
    # Test with start > end and positive step
    assert len(Range(10, 5)) == 0
    assert len(Range(10, 5, 1)) == 0
    
    # Test with start < end and negative step
    assert len(Range(5, 10, -1)) == 0
    
    # Test with step that doesn't evenly divide range
    assert len(Range(0, 10, 4)) == 2
    assert len(Range(1, 10, 3)) == 3


# LLM-generated content at query #16
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slicing with positive step
    r = Range(10)
    assert r[2:5] == [2, 3, 4]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slicing with negative indices
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slicing with step in slice
    assert r[2:8:2] == [2, 4, 6]
    assert r[1:9:3] == [1, 4, 7]
    
    # Test empty slices
    assert r[5:2] == []
    assert r[10:] == []
    
    # Test with negative step in Range constructor
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[9] == 1
    assert r[-1] == 1
    assert r[-10] == 10
    assert r[2:5] == [8, 7, 6]
    assert r[:3] == [10, 9, 8]
    
    # Test with larger step
    r = Range(0, 20, 3)
    assert r[0] == 0
    assert r[1] == 3
    assert r[6] == 18
    assert r[-1] == 18
    assert r[2:5] == [6, 9, 12]
    
    # Test single element slice
    assert r[3:4] == [9]
    
    # Test full slice
    assert r[:] == [0, 3, 6, 9, 12, 15, 18]
    
    # Test with negative start
    r = Range(-5, 5)
    assert r[0] == -5
    assert r[5] == 0
    assert r[9] == 4
    assert r[-1] == 4
    assert r[3:7] == [-2, -1, 0, 1]


# LLM-generated content at query #17
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test indexing with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[2] == 5
    assert r[3] == 7
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slice with positive step
    r = Range(10)
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slice with negative step
    r = Range(10)
    assert r[5:2:-1] == [5, 4, 3]
    assert r[9:6:-1] == [9, 8, 7]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert r[7:2:-2] == [7, 5, 3]
    
    # Test slice with start, stop, and step
    r = Range(5, 20, 3)
    assert r[:] == [5, 8, 11, 14, 17]
    assert r[1:4] == [8, 11, 14]
    assert r[::2] == [5, 11, 17]
    assert r[::-1] == [17, 14, 11, 8, 5]
    
    # Test slice with negative indices
    r = Range(10)
    assert r[-5:-2] == [5, 6, 7]
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-1:-6:-1] == [9, 8, 7, 6, 5]
    
    # Test empty slices
    r = Range(10)
    assert r[5:5] == []
    assert r[8:3] == []
    assert r[10:] == []
    assert r[5:3] == []
    
    # Test with negative step in constructor
    r = Range(10, 0, -2)
    assert r[0] == 10
    assert r[1] == 8
    assert r[2] == 6
    assert r[3] == 4
    assert r[4] == 2
    assert r[-1] == 2
    assert r[-2] == 4
    
    # Test slice with negative step in constructor
    r = Range(10, 0, -2)
    assert r[:] == [10, 8, 6, 4, 2]
    assert r[1:4] == [8, 6, 4]
    assert r[::-1] == [2, 4, 6, 8, 10]
    assert r[::2] == [10, 6, 2]
    
    # Test edge cases
    r = Range(0, 10, 3)
    assert r[0] == 0
    assert r[1] == 3
    assert r[2] == 6
    assert r[3] == 9
    assert r[-1] == 9
    assert r[-2] == 6
    
    # Test single element range
    r = Range(5, 6)
    assert r[0] == 5
    assert r[-1] == 5
    assert r[:] == [5]
    
    # Test with step that doesn't evenly divide range
    r = Range(0, 10, 4)
    assert r[0] == 0
    assert r[1] == 4
    assert r[2] == 8
    assert r[:] == [0, 4, 8]


# LLM-generated content at query #18
#--------------------------

```python
def test_LazyList___getitem__():
    # Test integer indexing
    lazy = LazyList(range(10))
    assert lazy[0] == 0
    assert lazy[5] == 5
    assert lazy[9] == 9
    
    # Test negative indexing
    lazy = LazyList(range(5))
    assert lazy[-1] == 4
    assert lazy[-2] == 3
    
    # Test slice indexing
    lazy = LazyList(range(10))
    assert lazy[2:5] == [2, 3, 4]
    assert lazy[:3] == [0, 1, 2]
    assert lazy[7:] == [7, 8, 9]
    assert lazy[::2] == [0, 2, 4, 6, 8]
    
    # Test that indexing triggers lazy evaluation
    lazy = LazyList(range(100))
    assert len(lazy.list) == 0
    _ = lazy[5]
    assert len(lazy.list) == 6
    _ = lazy[10]
    assert len(lazy.list) == 11
    
    # Test with exhausted iterator
    lazy = LazyList(range(3))
    _ = lazy[2]
    assert lazy.exhausted == True
    assert lazy[0] == 0
    assert lazy[1:3] == [1, 2]
    
    # Test with slice that goes beyond available elements
    lazy = LazyList(range(5))
    assert lazy[2:10] == [2, 3, 4]
    
    # Test with empty iterable
    lazy = LazyList([])
    try:
        _ = lazy[0]
        assert False, "Should raise IndexError"
    except IndexError:
        pass
    
    # Test with infinite iterator
    import itertools
    lazy = LazyList(itertools.count())
    assert lazy[0] == 0
    assert lazy[100] == 100
    assert lazy[50:55] == [50, 51, 52, 53, 54]
    
    # Test that negative slice stop works
    lazy = LazyList(range(10))
    assert lazy[2:-2] == [2, 3, 4, 5, 6, 7]
    
    # Test step in slice
    lazy = LazyList(range(20))
    assert lazy[2:10:2] == [2, 4, 6, 8]


# LLM-generated content at query #19
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test indexing with step
    r = Range(1, 11, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-5] == 1
    
    # Test slicing with positive step
    r = Range(10)
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Test slicing with negative indices
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slicing with step
    assert r[0:10:2] == [0, 2, 4, 6, 8]
    assert r[1:10:3] == [1, 4, 7]
    assert r[::2] == [0, 2, 4, 6, 8]
    
    # Test slicing with start, stop, step
    r = Range(5, 25, 3)
    assert r[:] == [5, 8, 11, 14, 17, 20, 23]
    assert r[2:5] == [11, 14, 17]
    assert r[::2] == [5, 11, 17, 23]
    assert r[1::2] == [8, 14, 20]
    
    # Test negative step (should work with slicing)
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[9] == 1
    assert r[:5] == [10, 9, 8, 7, 6]
    assert r[5:] == [5, 4, 3, 2, 1]
    
    # Test edge cases
    r = Range(3)
    assert r[0:0] == []
    assert r[5:10] == []
    assert r[100:200] == []
    
    # Test with single element range
    r = Range(1, 2)
    assert r[0] == 1
    assert r[:] == [1]
    assert r[-1] == 1
    
    # Test empty range
    r = Range(0)
    assert r[:] == []
    
    # Test range with negative start
    r = Range(-5, 5)
    assert r[0] == -5
    assert r[9] == 4
    assert r[-1] == 4
    assert r[3:7] == [-2, -1, 0, 1]
    
    # Test IndexError for out of bounds
    r = Range(10)
    try:
        r[10]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        r[-11]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test that slicing with negative step works correctly
    r = Range(10)
    assert r[9:0:-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert r[9:0:-2] == [9, 7, 5, 3, 1]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert r[5:1:-1] == [5, 4, 3, 2]


# LLM-generated content at query #20
#--------------------------

```python
def test_drop_until():
    # Test dropping until predicate is satisfied
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied at first element
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda c: c == 'l', 'hello')) == ['l', 'l', 'o']
    
    # Test with list of strings
    assert list(drop_until(lambda s: len(s) > 2, ['a', 'ab', 'abc', 'abcd'])) == ['abc', 'abcd']
    
    # Test with generator expression
    gen = (x for x in range(10) if x % 2 == 0)
    assert list(drop_until(lambda x: x > 4, gen)) == [6, 8]
    
    # Test with None values
    assert list(drop_until(lambda x: x is not None, [None, None, 1, 2, 3])) == [1, 2, 3]
    
    # Test with complex predicate
    assert list(drop_until(lambda x: x['value'] > 3, 
                          [{'value': 1}, {'value': 2}, {'value': 4}, {'value': 5}])) == \
                          [{'value': 4}, {'value': 5}]


# LLM-generated content at query #21
#--------------------------

```python
def test_LazyList___getitem__():
    # Test integer indexing
    lazy = LazyList(range(10))
    assert lazy[0] == 0
    assert lazy[5] == 5
    assert lazy[9] == 9
    
    # Test negative indexing
    lazy = LazyList(range(5))
    lazy._fetch_until(None)  # Exhaust the iterator
    assert lazy[-1] == 4
    assert lazy[-2] == 3
    
    # Test slice indexing
    lazy = LazyList(range(10))
    assert lazy[2:5] == [2, 3, 4]
    assert lazy[:3] == [0, 1, 2]
    assert lazy[7:] == [7, 8, 9]
    
    # Test slice with step
    lazy = LazyList(range(10))
    assert lazy[2:8:2] == [2, 4, 6]
    
    # Test that iterator is only consumed as needed
    lazy = LazyList(range(100))
    _ = lazy[5]  # Should only fetch up to index 5
    assert len(lazy.list) == 6  # 0-5 inclusive
    
    # Test slice that requires fetching more elements
    lazy = LazyList(range(10))
    result = lazy[3:7]
    assert result == [3, 4, 5, 6]
    assert len(lazy.list) == 7  # Should have fetched up to index 6
    
    # Test with infinite iterator
    import itertools
    lazy = LazyList(itertools.count())
    assert lazy[0] == 0
    assert lazy[100] == 100
    assert lazy[50:55] == [50, 51, 52, 53, 54]
    
    # Test IndexError for out of bounds on exhausted iterator
    lazy = LazyList(range(3))
    lazy._fetch_until(None)
    try:
        _ = lazy[5]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test that slice with negative stop doesn't trigger full fetch
    lazy = LazyList(range(10))
    # This should not fetch all elements since idx is None for negative indices
    result = lazy[2:-2]
    # The slice should work correctly after fetching all elements
    lazy._fetch_until(None)
    assert result == [2, 3, 4, 5, 6, 7]
    
    # Test empty slice
    lazy = LazyList(range(5))
    assert lazy[10:] == []
    
    # Test with non-integer iterable
    lazy = LazyList(["a", "b", "c", "d", "e"])
    assert lazy[2] == "c"
    assert lazy[1:4] == ["b", "c", "d"]


# LLM-generated content at query #22
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with single argument constructor
    r = Range(5)
    assert r[0] == 0
    assert r[1] == 1
    assert r[4] == 4
    assert r[-1] == 4
    assert r[-2] == 3
    
    # Test indexing with start and stop
    r = Range(2, 8)
    assert r[0] == 2
    assert r[1] == 3
    assert r[5] == 7
    assert r[-1] == 7
    assert r[-2] == 6
    
    # Test indexing with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[2] == 5
    assert r[3] == 7
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test negative step
    r = Range(10, 0, -2)
    assert r[0] == 10
    assert r[1] == 8
    assert r[2] == 6
    assert r[3] == 4
    assert r[4] == 2
    assert r[-1] == 2
    assert r[-2] == 4
    
    # Test slicing with single argument
    r = Range(10)
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert r[:5] == [0, 1, 2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[2:7] == [2, 3, 4, 5, 6]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slicing with start and stop
    r = Range(5, 15)
    assert r[:] == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert r[:3] == [5, 6, 7]
    assert r[5:] == [10, 11, 12, 13, 14]
    assert r[2:6] == [7, 8, 9, 10]
    assert r[::3] == [5, 8, 11, 14]
    
    # Test slicing with step
    r = Range(0, 20, 3)
    assert r[:] == [0, 3, 6, 9, 12, 15, 18]
    assert r[:3] == [0, 3, 6]
    assert r[3:] == [9, 12, 15, 18]
    assert r[1:4] == [3, 6, 9]
    assert r[::2] == [0, 6, 12, 18]
    
    # Test negative slicing
    r = Range(10)
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert r[5:2:-1] == [5, 4, 3]
    
    # Test slicing with negative step range
    r = Range(10, 0, -1)
    assert r[:] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert r[:3] == [10, 9, 8]
    assert r[3:] == [7, 6, 5, 4, 3, 2, 1]
    assert r[2:6] == [8, 7, 6, 5]
    assert r[::-1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Test edge cases
    r = Range(1)
    assert r[0] == 0
    assert r[:] == [0]
    assert r[-1] == 0
    
    r = Range(0)
    assert r[:] == []
    
    # Test with negative indices
    r = Range(1, 6)
    assert r[-1] == 5
    assert r[-2] == 4
    assert r[-3] == 3
    assert r[-4] == 2
    assert r[-5] == 1
    
    # Test slice with all parameters
    r = Range(20)
    assert r[2:10:2] == [2, 4, 6, 8]
    assert r[10:2:-2] == [10, 8, 6, 4]
    assert r[-1:-10:-2] == [19, 17, 15, 13, 11]
    
    # Test that indexing beyond bounds raises IndexError
    import pytest
    r = Range(5)
    with pytest.raises(IndexError):
        _ = r[5]
    with pytest.raises(IndexError):
        _ = r[-6]
    
    # Test that slicing beyond bounds works like list slicing
    r = Range(5)
    assert r[3:10] == [3, 4]
    assert r[-10:3] == [0, 1, 2]


# LLM-generated content at query #23
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[4] == 9
    assert r[-1] == 14
    assert r[-5] == 10
    
    # Test indexing with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[2] == 5
    assert r[3] == 7
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slicing with positive step
    r = Range(10)
    assert r[2:5] == [2, 3, 4]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slicing with start, stop, and step
    r = Range(5, 15)
    assert r[2:5] == [7, 8, 9]
    assert r[:3] == [5, 6, 7]
    assert r[5:] == [10, 11, 12, 13, 14]
    assert r[::2] == [5, 7, 9, 11, 13]
    
    # Test slicing with negative step
    r = Range(10)
    assert r[5:2:-1] == [5, 4, 3]
    assert r[9:6:-1] == [9, 8, 7]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert r[8:2:-2] == [8, 6, 4]
    
    # Test slicing with negative indices
    r = Range(10)
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    assert r[-1:-4:-1] == [9, 8, 7]
    
    # Test edge cases
    r = Range(5)
    assert r[:] == [0, 1, 2, 3, 4]
    assert r[0:0] == []
    assert r[3:1] == []
    assert r[100:200] == []
    
    # Test with negative step in constructor
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[5] == 5
    assert r[9] == 1
    assert r[-1] == 1
    assert r[-2] == 2
    
    # Test slicing with negative step in constructor
    r = Range(10, 0, -1)
    assert r[2:5] == [8, 7, 6]
    assert r[:3] == [10, 9, 8]
    assert r[5:] == [5, 4, 3, 2, 1]
    assert r[::2] == [10, 8, 6, 4, 2]
    assert r[::-1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Test with larger step
    r = Range(0, 20, 3)
    assert r[0] == 0
    assert r[1] == 3
    assert r[2] == 6
    assert r[-1] == 18
    assert r[-2] == 15
    assert r[2:5] == [6, 9, 12]
    assert r[::2] == [0, 6, 12, 18]


# LLM-generated content at query #24
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test indexing with step
    r = Range(1, 20, 3)
    assert r[0] == 1
    assert r[1] == 4
    assert r[2] == 7
    assert r[-1] == 19
    assert r[-2] == 16
    
    # Test slicing with positive step
    r = Range(10)
    assert r[0:5] == [0, 1, 2, 3, 4]
    assert r[2:8] == [2, 3, 4, 5, 6, 7]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Test slicing with negative indices
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slicing with step
    assert r[0:10:2] == [0, 2, 4, 6, 8]
    assert r[1:10:3] == [1, 4, 7]
    assert r[::2] == [0, 2, 4, 6, 8]
    
    # Test slicing with start, stop, and step
    r = Range(5, 25, 4)
    assert r[0:5] == [5, 9, 13, 17, 21]
    assert r[1:4] == [9, 13, 17]
    assert r[:3] == [5, 9, 13]
    assert r[2:] == [13, 17, 21]
    assert r[::2] == [5, 13, 21]
    
    # Test slicing with negative step
    assert r[::-1] == [21, 17, 13, 9, 5]
    assert r[4:1:-1] == [21, 17, 13]
    assert r[::-2] == [21, 13, 5]
    
    # Test edge cases
    r = Range(3)
    assert r[0:0] == []
    assert r[5:10] == []
    assert r[-10:-5] == []
    
    # Test with single element range
    r = Range(1, 2)
    assert r[0] == 1
    assert r[:] == [1]
    assert r[-1] == 1
    
    # Test with empty range
    r = Range(0)
    assert r[:] == []
    
    # Test with negative step
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[5] == 5
    assert r[9] == 1
    assert r[-1] == 1
    assert r[-10] == 10
    assert r[0:5] == [10, 9, 8, 7, 6]
    assert r[5:10] == [5, 4, 3, 2, 1]
    assert r[::-1] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Test with larger negative step
    r = Range(20, 0, -3)
    assert r[0] == 20
    assert r[1] == 17
    assert r[-1] == 2
    assert r[:] == [20, 17, 14, 11, 8, 5, 2]
    assert r[2:5] == [14, 11, 8]
    assert r[::-1] == [2, 5, 8, 11, 14, 17, 20]
    
    # Test IndexError for out of bounds
    r = Range(5)
    try:
        r[5]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        r[-6]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with step that doesn't evenly divide range
    r = Range(0, 10, 3)
    assert len(r) == 3
    assert r[0] == 0
    assert r[1] == 3
    assert r[2] == 6
    assert r[-1] == 6
    assert r[-2] == 3
    assert r[-3] == 0
    assert r[:] == [0, 3, 6]


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_MapList___getitem__():
    # Test integer indexing
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * 2, lst)
    assert map_list[0] == 2
    assert map_list[2] == 6
    assert map_list[-1] == 10
    assert map_list[-2] == 8

    # Test slice indexing
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x ** 2, lst)
    assert map_list[:] == [1, 4, 9, 16, 25]
    assert map_list[1:4] == [4, 9, 16]
    assert map_list[::2] == [1, 9, 25]
    assert map_list[::-1] == [25, 16, 9, 4, 1]

    # Test with empty list
    map_list = MapList(lambda x: x * 2, [])
    assert map_list[:] == []

    # Test with complex transformation
    lst = ["a", "bb", "ccc"]
    map_list = MapList(lambda s: len(s), lst)
    assert map_list[0] == 1
    assert map_list[1] == 2
    assert map_list[2] == 3
    assert map_list[:] == [1, 2, 3]

    # Test with Range as input
    map_list = MapList(lambda x: x * 3, Range(5))
    assert map_list[0] == 0
    assert map_list[2] == 6
    assert map_list[:3] == [0, 3, 6]

    # Test with another MapList as input
    base = MapList(lambda x: x + 1, [1, 2, 3])
    map_list = MapList(lambda x: x * 2, base)
    assert map_list[0] == 4
    assert map_list[1] == 6
    assert map_list[2] == 8
    assert map_list[:] == [4, 6, 8]

    # Test with negative slice indices
    lst = [10, 20, 30, 40, 50]
    map_list = MapList(lambda x: x // 10, lst)
    assert map_list[-3:] == [3, 4, 5]
    assert map_list[:-2] == [1, 2, 3]
    assert map_list[-4:-1] == [2, 3, 4]

    # Test with step in slice
    lst = list(range(10))
    map_list = MapList(lambda x: x * 2, lst)
    assert map_list[1:8:2] == [2, 6, 10, 14]
    assert map_list[::3] == [0, 6, 12, 18]


# LLM-generated content at query #2
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing with positive step
    r = Range(10)
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(1, 11)
    assert r[0] == 1
    assert r[9] == 10
    assert r[-1] == 10
    
    # Test indexing with step
    r = Range(1, 11, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slicing with positive step
    r = Range(10)
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert r[2:5] == [2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[:3] == [0, 1, 2]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slicing with negative indices
    r = Range(10)
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slicing with step in slice
    r = Range(10)
    assert r[2:8:2] == [2, 4, 6]
    assert r[::3] == [0, 3, 6, 9]
    assert r[1::3] == [1, 4, 7]
    
    # Test slicing with start, stop, and step
    r = Range(1, 11, 2)
    assert r[:] == [1, 3, 5, 7, 9]
    assert r[1:3] == [3, 5]
    assert r[::2] == [1, 5, 9]
    
    # Test edge cases
    r = Range(5)
    assert r[4] == 4
    assert r[-5] == 0
    
    # Test empty slices
    r = Range(10)
    assert r[5:5] == []
    assert r[10:] == []
    assert r[20:30] == []
    
    # Test with negative step in Range constructor
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[9] == 1
    assert r[:] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert r[2:5] == [8, 7, 6]
    
    # Test negative step with negative indices
    r = Range(10, 0, -1)
    assert r[-1] == 1
    assert r[-2] == 2
    assert r[-10] == 10
    
    # Test complex slicing with negative step
    r = Range(10, 0, -2)
    assert r[:] == [10, 8, 6, 4, 2]
    assert r[1:4] == [8, 6, 4]
    assert r[::2] == [10, 6, 2]


# LLM-generated content at query #3
#--------------------------

```python
def test_split_by():
    # Test with criterion
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]

    # Test with separator
    result = list(split_by("a.b.c", separator='.'))
    assert result == [['a'], ['b'], ['c']]

    # Test with empty_segments=True and separator
    result = list(split_by("..a..b..", empty_segments=True, separator='.'))
    assert result == [[], [], ['a'], [], ['b'], [], []]

    # Test with empty_segments=False and separator (default)
    result = list(split_by("..a..b..", separator='.'))
    assert result == [['a'], ['b']]

    # Test with empty iterable
    result = list(split_by([], criterion=lambda x: x))
    assert result == []

    # Test with empty_segments=True and empty iterable
    result = list(split_by([], empty_segments=True, criterion=lambda x: x))
    assert result == []

    # Test with no separators
    result = list(split_by([1, 2, 3], separator=0))
    assert result == [[1, 2, 3]]

    # Test with all separators
    result = list(split_by([0, 0, 0], separator=0))
    assert result == []

    # Test with all separators and empty_segments=True
    result = list(split_by([0, 0, 0], empty_segments=True, separator=0))
    assert result == [[], [], [], []]

    # Test with mixed types
    result = list(split_by([1, 'sep', 2, 'sep', 3], separator='sep'))
    assert result == [[1], [2], [3]]

    # Test error when both criterion and separator are specified
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x, separator=1))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test error when neither criterion nor separator are specified
    try:
        list(split_by([1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with criterion that drops all elements
    result = list(split_by([1, 2, 3], criterion=lambda x: True))
    assert result == []

    # Test with criterion that drops no elements
    result = list(split_by([1, 2, 3], criterion=lambda x: False))
    assert result == [[1, 2, 3]]

    # Test with strings and criterion
    result = list(split_by("hello world", criterion=lambda x: x == ' '))
    assert result == [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']]

    # Test with consecutive separators
    result = list(split_by([1, 0, 0, 2, 0, 3], separator=0))
    assert result == [[1], [2], [3]]


# LLM-generated content at query #4
#--------------------------

```python
def test_MapList___getitem__():
    # Test integer indexing
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * 2, lst)
    assert map_list[0] == 2
    assert map_list[2] == 6
    assert map_list[-1] == 10
    assert map_list[-2] == 8

    # Test slice indexing
    assert map_list[1:3] == [4, 6]
    assert map_list[:3] == [2, 4, 6]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[::2] == [2, 6, 10]
    assert map_list[::-1] == [10, 8, 6, 4, 2]

    # Test with different transformation
    map_list2 = MapList(lambda x: x ** 2, lst)
    assert map_list2[0] == 1
    assert map_list2[2] == 9
    assert map_list2[1:4] == [4, 9, 16]

    # Test with empty list
    empty_map = MapList(lambda x: x * 2, [])
    with pytest.raises(IndexError):
        _ = empty_map[0]
    assert empty_map[:] == []

    # Test with Range as input
    range_map = MapList(lambda x: x + 10, Range(5))
    assert range_map[0] == 10
    assert range_map[3] == 13
    assert range_map[1:4] == [11, 12, 13]

    # Test with string transformation
    str_map = MapList(lambda x: str(x) + "!", ["a", "b", "c"])
    assert str_map[0] == "a!"
    assert str_map[1:3] == ["b!", "c!"]

    # Test with complex transformation
    complex_map = MapList(lambda x: (x, x**2), [1, 2, 3])
    assert complex_map[0] == (1, 1)
    assert complex_map[1] == (2, 4)
    assert complex_map[:2] == [(1, 1), (2, 4)]


# LLM-generated content at query #5
#--------------------------

```python
def test_split_by():
    # Test with criterion
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]
    
    # Test with separator
    result = list(split_by("a.b.c", separator='.'))
    assert result == [['a'], ['b'], ['c']]
    
    # Test with empty_segments=True and separator
    result = list(split_by(" Split by: ", empty_segments=True, separator='.'))
    assert result == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]
    
    # Test with empty iterable
    result = list(split_by([], criterion=lambda x: x))
    assert result == []
    
    # Test with empty_segments=True and empty iterable
    result = list(split_by([], empty_segments=True, criterion=lambda x: x))
    assert result == []
    
    # Test with all elements satisfying criterion
    result = list(split_by([1, 2, 3], criterion=lambda x: True))
    assert result == []
    
    # Test with all elements satisfying criterion and empty_segments=True
    result = list(split_by([1, 2, 3], empty_segments=True, criterion=lambda x: True))
    assert result == [[], [], [], []]
    
    # Test with no elements satisfying criterion
    result = list(split_by([1, 2, 3], criterion=lambda x: False))
    assert result == [[1, 2, 3]]
    
    # Test with consecutive separators
    result = list(split_by("a..b", separator='.'))
    assert result == [['a'], ['b']]
    
    # Test with consecutive separators and empty_segments=True
    result = list(split_by("a..b", empty_segments=True, separator='.'))
    assert result == [['a'], [], ['b']]
    
    # Test with separator at beginning and end
    result = list(split_by(".a.b.", separator='.'))
    assert result == [['a'], ['b']]
    
    # Test with separator at beginning and end with empty_segments=True
    result = list(split_by(".a.b.", empty_segments=True, separator='.'))
    assert result == [[], ['a'], ['b'], []]
    
    # Test error when both criterion and separator are None
    try:
        list(split_by([1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test error when both criterion and separator are specified
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x, separator=1))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with complex criterion
    result = list(split_by([1, 2, 3, 4, 5], criterion=lambda x: x % 2 == 0))
    assert result == [[1], [3], [5]]
    
    # Test with string separator in list of strings
    result = list(split_by(["a", "sep", "b", "sep", "c"], separator="sep"))
    assert result == [['a'], ['b'], ['c']]


# LLM-generated content at query #6
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied immediately
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda c: c == 'b', 'abcdef')) == ['b', 'c', 'd', 'e', 'f']
    
    # Test with list
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7, 8])) == [6, 7, 8]
    
    # Test with generator
    gen = (x for x in range(10))
    assert list(drop_until(lambda x: x > 7, gen)) == [8, 9]
    
    # Test with single element that satisfies predicate
    assert list(drop_until(lambda x: x == 5, [5])) == [5]
    
    # Test with single element that doesn't satisfy predicate
    assert list(drop_until(lambda x: x == 5, [3])) == []
    
    # Test with predicate satisfied at last element
    assert list(drop_until(lambda x: x == 9, range(10))) == [9]
    
    # Test with all elements satisfying predicate
    assert list(drop_until(lambda x: True, [1, 2, 3])) == [1, 2, 3]
    
    # Test with no elements satisfying predicate
    assert list(drop_until(lambda x: False, [1, 2, 3])) == []


# LLM-generated content at query #7
#--------------------------

```python
def test_split_by():
    # Test with criterion
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]
    
    # Test with separator
    result = list(split_by("a.b.c", separator='.'))
    assert result == [['a'], ['b'], ['c']]
    
    # Test with empty_segments=True and separator
    result = list(split_by(" Split by: ", empty_segments=True, separator=' '))
    assert result == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]
    
    # Test with empty iterable
    result = list(split_by([], criterion=lambda x: x))
    assert result == []
    
    # Test with empty_segments=True and empty iterable
    result = list(split_by([], empty_segments=True, criterion=lambda x: x))
    assert result == []
    
    # Test with all elements satisfying criterion
    result = list(split_by([1, 2, 3], criterion=lambda x: True))
    assert result == []
    
    # Test with empty_segments=True and all elements satisfying criterion
    result = list(split_by([1, 2, 3], empty_segments=True, criterion=lambda x: True))
    assert result == [[], [], [], []]
    
    # Test with no elements satisfying criterion
    result = list(split_by([1, 2, 3], criterion=lambda x: False))
    assert result == [[1, 2, 3]]
    
    # Test with mixed types and separator
    result = list(split_by([1, 'sep', 2, 'sep', 3], separator='sep'))
    assert result == [[1], [2], [3]]
    
    # Test error when both criterion and separator are specified
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x > 1, separator=2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test error when neither criterion nor separator are specified
    try:
        list(split_by([1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with string separator and empty segments
    result = list(split_by("...", empty_segments=True, separator='.'))
    assert result == [[], [], [], []]
    
    # Test with consecutive separators
    result = list(split_by("a..b", separator='.'))
    assert result == [['a'], ['b']]
    
    # Test with trailing separator
    result = list(split_by("a.b.", separator='.'))
    assert result == [['a'], ['b']]
    
    # Test with leading separator
    result = list(split_by(".a.b", separator='.'))
    assert result == [['a'], ['b']]


# LLM-generated content at query #8
#--------------------------

```python
def test_chunk():
    # Test basic chunking
    assert list(chunk(3, range(10))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    
    # Test exact division
    assert list(chunk(2, range(6))) == [[0, 1], [2, 3], [4, 5]]
    
    # Test chunk size larger than iterable
    assert list(chunk(5, range(3))) == [[0, 1, 2]]
    
    # Test chunk size of 1
    assert list(chunk(1, range(3))) == [[0], [1], [2]]
    
    # Test empty iterable
    assert list(chunk(3, [])) == []
    
    # Test with non-range iterable
    assert list(chunk(2, "abcdef")) == [["a", "b"], ["c", "d"], ["e", "f"]]
    
    # Test with generator
    assert list(chunk(2, (x for x in range(5)))) == [[0, 1], [2, 3], [4]]
    
    # Test error for non-positive n
    try:
        list(chunk(0, range(5)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        list(chunk(-1, range(5)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_LazyList___getitem__():
    # Test integer indexing
    lazy = LazyList(range(10))
    assert lazy[0] == 0
    assert lazy[5] == 5
    assert lazy[9] == 9
    
    # Test negative indexing
    lazy = LazyList(range(5))
    lazy._fetch_until(None)  # Exhaust the iterator
    assert lazy[-1] == 4
    assert lazy[-2] == 3
    
    # Test slice indexing
    lazy = LazyList(range(10))
    assert lazy[2:5] == [2, 3, 4]
    assert lazy[:3] == [0, 1, 2]
    assert lazy[7:] == [7, 8, 9]
    assert lazy[::2] == [0, 2, 4, 6, 8]
    
    # Test that iterator is only consumed as needed
    lazy = LazyList(range(100))
    _ = lazy[10]  # Should only fetch up to index 10
    assert len(lazy.list) == 11  # 0-10 inclusive
    assert not lazy.exhausted
    
    # Test with exhausted iterator
    lazy = LazyList(range(5))
    lazy._fetch_until(None)
    assert lazy.exhausted
    assert lazy[3] == 3
    assert lazy[1:4] == [1, 2, 3]
    
    # Test with infinite iterator (simulated)
    def infinite():
        i = 0
        while True:
            yield i
            i += 1
    
    lazy = LazyList(infinite())
    assert lazy[0] == 0
    assert lazy[100] == 100
    assert lazy[50:55] == [50, 51, 52, 53, 54]
    
    # Test IndexError for out of bounds
    lazy = LazyList(range(5))
    try:
        _ = lazy[10]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test that slice with None stop fetches everything
    lazy = LazyList(range(5))
    _ = lazy[:]
    assert lazy.exhausted
    assert lazy.list == [0, 1, 2, 3, 4]
    
    # Test with empty iterable
    lazy = LazyList([])
    try:
        _ = lazy[0]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test slice with negative indices
    lazy = LazyList(range(10))
    lazy._fetch_until(None)
    assert lazy[-3:] == [7, 8, 9]
    assert lazy[-5:-2] == [5, 6, 7]
    
    # Test that __getitem__ doesn't modify original list
    lazy = LazyList([1, 2, 3, 4, 5])
    result = lazy[2]
    assert result == 3
    assert lazy.list == [1, 2, 3]  # Only fetched up to index 2
    
    # Test with step in slice
    lazy = LazyList(range(20))
    assert lazy[2:10:2] == [2, 4, 6, 8]
    assert lazy[::3] == [0, 3, 6, 9, 12, 15, 18]


# LLM-generated content at query #10
#--------------------------

```python
def test_split_by():
    # Test with criterion
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]
    
    # Test with separator
    result = list(split_by("a.b.c", separator='.'))
    assert result == [['a'], ['b'], ['c']]
    
    # Test with empty_segments=True and separator
    result = list(split_by("..a..b..", empty_segments=True, separator='.'))
    assert result == [[], [], ['a'], [], ['b'], [], []]
    
    # Test with empty_segments=False and separator (default)
    result = list(split_by("..a..b..", separator='.'))
    assert result == [['a'], ['b']]
    
    # Test with empty iterable
    result = list(split_by([], criterion=lambda x: x))
    assert result == []
    
    # Test with empty iterable and empty_segments=True
    result = list(split_by([], empty_segments=True, criterion=lambda x: x))
    assert result == []
    
    # Test with no separators
    result = list(split_by([1, 2, 3], separator=0))
    assert result == [[1, 2, 3]]
    
    # Test with all separators
    result = list(split_by([0, 0, 0], empty_segments=True, separator=0))
    assert result == [[], [], [], []]
    
    # Test with all separators and empty_segments=False
    result = list(split_by([0, 0, 0], separator=0))
    assert result == []
    
    # Test with string and criterion
    result = list(split_by("hello world", criterion=lambda x: x == ' '))
    assert result == [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']]
    
    # Test error when both criterion and separator are specified
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x > 1, separator=2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test error when neither criterion nor separator are specified
    try:
        list(split_by([1, 2, 3]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_Range___getitem__():
    # Test single index access
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slice access
    r = Range(10)
    assert r[2:5] == [2, 3, 4]
    assert r[:3] == [0, 1, 2]
    assert r[7:] == [7, 8, 9]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slice with negative step
    r = Range(10)
    assert r[5:2:-1] == [5, 4, 3]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert r[8:2:-2] == [8, 6, 4]
    
    # Test slice with start, stop, step
    r = Range(1, 20, 3)
    assert r[2:5] == [7, 10, 13]
    assert r[:3] == [1, 4, 7]
    assert r[3:] == [10, 13, 16, 19]
    assert r[::2] == [1, 7, 13, 19]
    
    # Test empty slices
    r = Range(10)
    assert r[5:5] == []
    assert r[10:] == []
    assert r[5:3] == []
    
    # Test with negative indices in slices
    r = Range(10)
    assert r[-3:-1] == [7, 8]
    assert r[-5:] == [5, 6, 7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    
    # Test edge cases
    r = Range(1)
    assert r[0] == 0
    assert r[:] == [0]
    
    r = Range(0)
    assert r[:] == []
    
    # Test with larger step
    r = Range(0, 100, 10)
    assert r[5] == 50
    assert r[-1] == 90
    assert r[2:5] == [20, 30, 40]
    
    # Test with negative step in constructor
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[9] == 1
    assert r[-1] == 1
    assert r[2:5] == [8, 7, 6]


# LLM-generated content at query #12
#--------------------------

```python
def test_LazyList___getitem__():
    # Test integer indexing
    lazy = LazyList(range(10))
    assert lazy[0] == 0
    assert lazy[5] == 5
    assert lazy[9] == 9
    
    # Test negative indexing
    lazy = LazyList(range(5))
    assert lazy[-1] == 4
    assert lazy[-2] == 3
    
    # Test slice indexing
    lazy = LazyList(range(10))
    assert lazy[2:5] == [2, 3, 4]
    assert lazy[:3] == [0, 1, 2]
    assert lazy[7:] == [7, 8, 9]
    assert lazy[::2] == [0, 2, 4, 6, 8]
    
    # Test that indexing triggers fetching
    lazy = LazyList(range(100))
    _ = lazy[50]
    assert len(lazy.list) == 51
    
    # Test with exhausted iterator
    lazy = LazyList(range(3))
    _ = lazy[0]
    _ = lazy[1]
    _ = lazy[2]
    assert lazy.exhausted == True
    assert lazy[0:2] == [0, 1]
    
    # Test with slice that goes beyond available elements
    lazy = LazyList(range(5))
    result = lazy[2:10]
    assert result == [2, 3, 4]
    
    # Test with empty iterable
    lazy = LazyList([])
    try:
        _ = lazy[0]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test that multiple accesses work correctly
    lazy = LazyList(range(5))
    assert lazy[2] == 2
    assert lazy[1] == 1
    assert lazy[3] == 3
    
    # Test with infinite iterator (simulated)
    def infinite():
        i = 0
        while True:
            yield i
            i += 1
    
    lazy = LazyList(infinite())
    assert lazy[0] == 0
    assert lazy[100] == 100
    assert lazy[50:55] == [50, 51, 52, 53, 54]


# LLM-generated content at query #13
#--------------------------

```python
def test_drop_until():
    # Test dropping until predicate is satisfied
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: x > 5, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied at first element
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda x: x == 'b', ['a', 'b', 'c', 'd'])) == ['b', 'c', 'd']
    
    # Test with list of booleans
    assert list(drop_until(lambda x: x, [False, False, True, False, True])) == [True, False, True]
    
    # Test with custom objects
    class TestObj:
        def __init__(self, val):
            self.val = val
        def __eq__(self, other):
            return self.val == other.val
    
    objs = [TestObj(1), TestObj(2), TestObj(3)]
    result = list(drop_until(lambda x: x.val > 1, objs))
    assert len(result) == 2
    assert result[0].val == 2
    assert result[1].val == 3
    
    # Test with generator
    gen = (x for x in range(10))
    assert list(drop_until(lambda x: x > 7, gen)) == [8, 9]
    
    # Test with predicate that checks type
    mixed = [1, 'a', 2, 'b', 3]
    assert list(drop_until(lambda x: isinstance(x, str), mixed)) == ['a', 2, 'b', 3]
    
    # Test where predicate is satisfied at last element
    assert list(drop_until(lambda x: x == 4, range(5))) == [4]
    
    # Test with negative numbers
    assert list(drop_until(lambda x: x >= 0, [-5, -3, -1, 0, 2, 4])) == [0, 2, 4]


# LLM-generated content at query #14
#--------------------------

```python
def test_drop_until():
    # Test basic functionality - drop until predicate is satisfied
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]
    
    # Test when predicate is satisfied at first element
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]
    
    # Test when predicate is never satisfied
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []
    
    # Test with empty iterable
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []
    
    # Test with string iterable
    result = list(drop_until(lambda x: x == 'b', 'abcdef'))
    assert result == ['b', 'c', 'd', 'e', 'f']
    
    # Test with list of strings
    result = list(drop_until(lambda x: len(x) > 2, ['a', 'ab', 'abc', 'abcd']))
    assert result == ['abc', 'abcd']
    
    # Test with iterator (not just list)
    result = list(drop_until(lambda x: x % 3 == 0, iter([1, 2, 3, 4, 5, 6])))
    assert result == [3, 4, 5, 6]
    
    # Test with complex predicate
    result = list(drop_until(lambda x: x['value'] > 3, 
                            [{'value': 1}, {'value': 2}, {'value': 3}, {'value': 4}]))
    assert result == [{'value': 4}]
    
    # Test that it works with infinite generators (partial consumption)
    from itertools import count
    result = list(take(3, drop_until(lambda x: x > 5, count())))
    assert result == [6, 7, 8]
    
    # Test with None values
    result = list(drop_until(lambda x: x is not None, [None, None, 1, 2, 3]))
    assert result == [1, 2, 3]


# LLM-generated content at query #15
#--------------------------

```python
def test_Range___getitem__():
    # Test single index access
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[5] == 10
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slice access
    r = Range(10)
    assert r[:] == list(range(10))
    assert r[2:5] == [2, 3, 4]
    assert r[5:2:-1] == [5, 4, 3]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    
    # Test slice with start and stop
    r = Range(5, 15)
    assert r[2:5] == [7, 8, 9]
    assert r[:3] == [5, 6, 7]
    assert r[5:] == [10, 11, 12, 13, 14]
    
    # Test slice with step
    r = Range(1, 10, 2)
    assert r[:] == [1, 3, 5, 7, 9]
    assert r[1:3] == [3, 5]
    assert r[::2] == [1, 5, 9]
    
    # Test edge cases
    r = Range(0, 10, 3)
    assert r[0] == 0
    assert r[1] == 3
    assert r[2] == 6
    assert r[3] == 9
    assert r[-1] == 9
    assert r[-2] == 6
    
    # Test empty range
    r = Range(0)
    assert r[:] == []
    
    # Test single element range
    r = Range(5, 6)
    assert r[0] == 5
    assert r[-1] == 5
    assert r[:] == [5]
    
    # Test with negative step
    r = Range(10, 0, -1)
    assert r[0] == 10
    assert r[5] == 5
    assert r[9] == 1
    assert r[-1] == 1
    assert r[-10] == 10
    assert r[:] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Test slice with negative step
    r = Range(10, 0, -2)
    assert r[:] == [10, 8, 6, 4, 2]
    assert r[1:3] == [8, 6]
    assert r[::-1] == [2, 4, 6, 8, 10]


# LLM-generated content at query #16
#--------------------------

```python
def test_Range___getitem__():
    # Test basic indexing with positive step
    r = Range(10)
    assert r[0] == 0
    assert r[5] == 5
    assert r[9] == 9
    
    # Test negative indexing
    assert r[-1] == 9
    assert r[-2] == 8
    assert r[-10] == 0
    
    # Test indexing with start and stop
    r = Range(5, 15)
    assert r[0] == 5
    assert r[9] == 14
    assert r[-1] == 14
    assert r[-10] == 5
    
    # Test indexing with step
    r = Range(1, 10, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9
    assert r[-1] == 9
    assert r[-2] == 7
    
    # Test slicing with positive step
    r = Range(10)
    assert r[:] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert r[2:5] == [2, 3, 4]
    assert r[5:] == [5, 6, 7, 8, 9]
    assert r[:3] == [0, 1, 2]
    assert r[::2] == [0, 2, 4, 6, 8]
    assert r[1::2] == [1, 3, 5, 7, 9]
    
    # Test slicing with negative indices
    assert r[-3:] == [7, 8, 9]
    assert r[:-3] == [0, 1, 2, 3, 4, 5, 6]
    assert r[-5:-2] == [5, 6, 7]
    
    # Test slicing with step in slice
    assert r[1:8:2] == [1, 3, 5, 7]
    assert r[8:1:-2] == [8, 6, 4, 2]
    assert r[::-1] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    # Test slicing with start, stop, and step
    r = Range(1, 11, 2)
    assert r[:] == [1, 3, 5, 7, 9]
    assert r[1:3] == [3, 5]
    assert r[::2] == [1, 5, 9]
    assert r[::-1] == [9, 7, 5, 3, 1]
    
    # Test edge cases
    r = Range(5)
    assert r[4] == 4
    assert r[-5] == 0
    
    # Test with negative step (though Range doesn't support negative step in constructor)
    # Note: Range(10, 0, -1) would be valid but not implemented in the given code
    # The implementation only supports positive step
    
    # Test empty range
    r = Range(0)
    assert len(r) == 0
    # Indexing empty range should raise IndexError
    try:
        r[0]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test single element range
    r = Range(1, 2)
    assert r[0] == 1
    assert r[-1] == 1
    assert r[:] == [1]
    
    # Test that __getitem__ works with slice objects that have None values
    r = Range(10)
    assert r[None:None] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert r[None:5] == [0, 1, 2, 3, 4]
    assert r[5:None] == [5, 6, 7, 8, 9]


# LLM-generated content at query #17
#--------------------------

```python
def test_drop_until():
    # Test basic functionality
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]
    
    # Test with empty iterable
    assert list(drop_until(lambda x: True, [])) == []
    
    # Test where predicate is never satisfied
    assert list(drop_until(lambda x: x > 10, range(5))) == []
    
    # Test where predicate is satisfied immediately
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    assert list(drop_until(lambda c: c == 'l', "hello world")) == ['l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    
    # Test with list
    assert list(drop_until(lambda x: x % 2 == 0, [1, 3, 5, 6, 7, 8])) == [6, 7, 8]
    
    # Test with iterator
    assert list(drop_until(lambda x: x == 3, iter([1, 2, 3, 4, 5]))) == [3, 4, 5]
    
    # Test with all elements satisfying predicate
    assert list(drop_until(lambda x: True, [1, 2, 3])) == [1, 2, 3]
    
    # Test with none elements satisfying predicate
    assert list(drop_until(lambda x: False, [1, 2, 3])) == []
    
    # Test with complex predicate
    assert list(drop_until(lambda x: len(x) > 3, ["a", "ab", "abc", "abcd", "abcde"])) == ["abcd", "abcde"]


# LLM-generated content at query #18
#--------------------------

```python
def test_drop_until():
    # Test dropping until predicate is satisfied
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]
    
    # Test with empty iterable
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []
    
    # Test where predicate is never satisfied
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []
    
    # Test where first element satisfies predicate
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]
    
    # Test with string iterable
    result = list(drop_until(lambda c: c == 'l', 'hello world'))
    assert result == ['l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    
    # Test with list of strings
    result = list(drop_until(lambda s: len(s) > 3, ['a', 'ab', 'abc', 'abcd', 'abcde']))
    assert result == ['abcd', 'abcde']
    
    # Test with generator expression
    result = list(drop_until(lambda x: x % 2 == 0, (x for x in range(10))))
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Test where predicate is satisfied after some elements
    result = list(drop_until(lambda x: x == 3, [1, 2, 3, 4, 5]))
    assert result == [3, 4, 5]
    
    # Test with single element iterable that satisfies predicate
    result = list(drop_until(lambda x: x == 1, [1]))
    assert result == [1]
    
    # Test with single element iterable that doesn't satisfy predicate
    result = list(drop_until(lambda x: x == 2, [1]))
    assert result == []


