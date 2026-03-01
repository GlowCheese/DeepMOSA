####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_range_iter_single_arg():
    r = Range(5)
    assert list(iter(r)) == [0, 1, 2, 3, 4]

def test_range_iter_two_args():
    r = Range(2, 6)
    assert list(iter(r)) == [2, 3, 4, 5]

def test_range_iter_three_args():
    r = Range(1, 10, 2)
    assert list(iter(r)) == [1, 3, 5, 7, 9]

def test_range_iter_empty():
    r = Range(0)
    assert list(iter(r)) == []

def test_range_iter_negative_step():
    r = Range(5, 0, -1)
    assert list(iter(r)) == [5, 4, 3, 2, 1]

def test_range_iter_multiple_iterations():
    r = Range(3)
    assert list(iter(r)) == [0, 1, 2]
    assert list(iter(r)) == [0, 1, 2]


# LLM-generated content at query #2
#--------------------------

```python
def test_single_index_access():
    r = Range(1, 11, 2)
    assert r[0] == 1
    assert r[1] == 3
    assert r[4] == 9

def test_negative_index_access():
    r = Range(1, 11, 2)
    assert r[-1] == 9
    assert r[-2] == 7
    assert r[-5] == 1

def test_slice_access():
    r = Range(1, 11, 2)
    assert r[1:4] == [3, 5, 7]
    assert r[:3] == [1, 3, 5]
    assert r[2:] == [5, 7, 9]
    assert r[:] == [1, 3, 5, 7, 9]

def test_empty_slice():
    r = Range(1, 11, 2)
    assert r[10:20] == []
    assert r[-10:-20] == []

def test_step_in_slice():
    r = Range(1, 11, 2)
    assert r[::2] == [1, 5, 9]
    assert r[1::2] == [3, 7]
    assert r[::-1] == [9, 7, 5, 3, 1]


# LLM-generated content at query #3
#--------------------------

```python
def test_lazy_list_constructor():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #4
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    lst = LazyList([1, 2, 3])
    assert lst.exhausted is False
    assert lst.list == []
    assert hasattr(lst, 'iter')


# LLM-generated content at query #5
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')

def test_lazy_list_constructor_with_generator():
    def gen():
        yield 1
        yield 2
    lazy_list = LazyList(gen())
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #6
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_iterable_is_converted_to_iterator():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert isinstance(lazy_list.iter, Iterator)


# LLM-generated content at query #8
#--------------------------

```python
def test_constructor_with_valid_inputs():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #9
#--------------------------

```python
def test_scanl_with_initial_value():
    result = list(scanl(lambda x, y: x + y, [1, 2, 3], 0))
    assert result == [0, 1, 3, 6]

def test_scanl_without_initial_value():
    result = list(scanl(lambda x, y: x + y, [1, 2, 3]))
    assert result == [1, 3, 6]

def test_scanl_with_string_concatenation():
    result = list(scanl(lambda x, y: y + x, ['a', 'b', 'c']))
    assert result == ['a', 'ba', 'cba']

def test_scanl_with_multiplication():
    result = list(scanl(lambda x, y: x * y, [2, 3, 4], 1))
    assert result == [1, 2, 6, 24]

def test_scanl_with_empty_iterable_and_initial():
    result = list(scanl(lambda x, y: x + y, [], 5))
    assert result == [5]

def test_scanl_with_single_element_and_initial():
    result = list(scanl(lambda x, y: x + y, [10], 5))
    assert result == [5, 15]

def test_scanl_with_single_element_no_initial():
    result = list(scanl(lambda x, y: x + y, [10]))
    assert result == [10]

def test_scanl_with_too_many_args():
    try:
        list(scanl(lambda x, y: x + y, [1, 2], 0, 1))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.val == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.val == 1
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_split_by_criterion_without_empty_segments():
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]

def test_split_by_criterion_with_empty_segments():
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0, empty_segments=True))
    assert result == [[1, 2], [4, 5], [7, 8], []]

def test_split_by_separator_without_empty_segments():
    result = list(split_by("Split by:", separator=' '))
    assert result == [['S', 'p', 'l', 'i', 't'], ['b', 'y:']]

def test_split_by_separator_with_empty_segments():
    result = list(split_by(" Split by: ", empty_segments=True, separator=' '))
    assert result == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []]

def test_split_by_empty_iterable():
    result = list(split_by([], criterion=lambda x: x % 3 == 0))
    assert result == [[]]

def test_split_by_empty_iterable_with_empty_segments():
    result = list(split_by([], criterion=lambda x: x % 3 == 0, empty_segments=True))
    assert result == [[]]

def test_split_by_all_elements_match_criterion():
    result = list(split_by([3, 6, 9], criterion=lambda x: x % 3 == 0))
    assert result == []

def test_split_by_all_elements_match_criterion_with_empty_segments():
    result = list(split_by([3, 6, 9], criterion=lambda x: x % 3 == 0, empty_segments=True))
    assert result == [[], [], []]

def test_split_by_no_elements_match_criterion():
    result = list(split_by([1, 2, 4], criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2, 4]]

def test_split_by_invalid_parameters():
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x % 3 == 0, separator=' '))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_split_by_none_criterion_and_separator():
    try:
        list(split_by([1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_drop_first_n_elements():
    assert list(drop(3, [1, 2, 3, 4, 5])) == [4, 5]

def test_drop_zero_elements():
    assert list(drop(0, [1, 2, 3])) == [1, 2, 3]

def test_drop_all_elements():
    assert list(drop(5, [1, 2, 3])) == []

def test_drop_more_elements_than_available():
    assert list(drop(10, [1, 2, 3])) == []

def test_drop_with_empty_iterable():
    assert list(drop(5, [])) == []

def test_drop_with_generator():
    gen = (x for x in range(10))
    assert list(drop(7, gen)) == [7, 8, 9]

def test_drop_with_string():
    assert list(drop(2, "hello")) == ['l', 'l', 'o']

def test_drop_negative_n_raises_value_error():
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    iterable = [1, 2, 3, 4, 5]
    lazy_list = LazyList(iterable)

    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert isinstance(lazy_list.iter, type(iter(iterable)))


# LLM-generated content at query #14
#--------------------------

```python
def test_take_first_n_elements():
    assert list(take(5, range(1000000))) == [0, 1, 2, 3, 4]

def test_take_zero_elements():
    assert list(take(0, range(10))) == []

def test_take_more_elements_than_iterable():
    assert list(take(10, range(5))) == [0, 1, 2, 3, 4]

def test_take_negative_n_raises_error():
    try:
        list(take(-1, range(10)))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_take_with_empty_iterable():
    assert list(take(5, [])) == []

def test_take_with_string_iterable():
    assert list(take(3, "hello")) == ['h', 'e', 'l']

def test_take_with_list_iterable():
    assert list(take(2, [1, 2, 3, 4])) == [1, 2]


# LLM-generated content at query #15
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.val == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.val == 1
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_chunk_empty_iterable():
    assert list(chunk(3, [])) == []

def test_chunk_exact_division():
    assert list(chunk(3, [1, 2, 3, 4, 5, 6])) == [[1, 2, 3], [4, 5, 6]]

def test_chunk_uneven_division():
    assert list(chunk(3, [1, 2, 3, 4, 5, 6, 7])) == [[1, 2, 3], [4, 5, 6], [7]]

def test_chunk_single_element():
    assert list(chunk(1, [1, 2, 3])) == [[1], [2], [3]]

def test_chunk_large_chunk_size():
    assert list(chunk(10, [1, 2, 3])) == [[1, 2, 3]]

def test_chunk_with_range():
    assert list(chunk(3, range(10))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

def test_chunk_raises_value_error_for_non_positive_n():
    try:
        list(chunk(0, [1, 2, 3]))
    except ValueError as e:
        assert str(e) == "`n` should be positive"
    else:
        assert False, "Expected ValueError"

def test_chunk_raises_value_error_for_negative_n():
    try:
        list(chunk(-1, [1, 2, 3]))
    except ValueError as e:
        assert str(e) == "`n` should be positive"
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #18
#--------------------------

```python
def test_getitem_single_index_within_range():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[0] == 1
    assert ll[2] == 3
    assert ll[4] == 5

def test_getitem_single_index_out_of_range():
    ll = LazyList([1, 2, 3])
    try:
        _ = ll[5]
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_getitem_negative_index():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[-1] == 5
    assert ll[-3] == 3

def test_getitem_slice_full():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[:] == [1, 2, 3, 4, 5]

def test_getitem_slice_start_only():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[2:] == [3, 4, 5]

def test_getitem_slice_stop_only():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[:3] == [1, 2, 3]

def test_getitem_slice_start_and_stop():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[1:4] == [2, 3, 4]

def test_getitem_slice_with_step():
    ll = LazyList([1, 2, 3, 4, 5])
    assert ll[::2] == [1, 3, 5]
    assert ll[1::2] == [2, 4]

def test_getitem_lazy_behavior():
    ll = LazyList(range(100))
    assert ll[0] == 0
    assert ll[50] == 50
    assert len(ll.list) == 51
    assert ll[99] == 99
    assert ll.exhausted is True


# LLM-generated content at query #19
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10
    assert r.val == 0

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10
    assert r.val == 1

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5
    assert r.val == 1

def test_range_constructor_zero_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_four_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_getitem_single_index():
    func = lambda x: x * 2
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(func, lst)
    assert map_list[0] == 2
    assert map_list[1] == 4
    assert map_list[2] == 6
    assert map_list[3] == 8
    assert map_list[4] == 10

def test_getitem_slice():
    func = lambda x: x * 2
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(func, lst)
    assert map_list[1:3] == [4, 6]
    assert map_list[:2] == [2, 4]
    assert map_list[2:] == [6, 8, 10]
    assert map_list[:] == [2, 4, 6, 8, 10]
    assert map_list[::2] == [2, 6, 10]

def test_getitem_negative_index():
    func = lambda x: x * 2
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(func, lst)
    assert map_list[-1] == 10
    assert map_list[-2] == 8
    assert map_list[-3] == 6


# LLM-generated content at query #21
#--------------------------

```python
def test_lazy_list_constructor_initializes_correctly():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.exhausted is False
    assert lazy_list.list == []
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #22
#--------------------------

```python
def test_getitem_with_int_index():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    mapped_list = MapList(func, lst)
    result = mapped_list[1]
    assert result == 4


# LLM-generated content at query #23
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_getitem_with_int_index():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list[0] == 2
    assert map_list[1] == 4
    assert map_list[2] == 6


# LLM-generated content at query #25
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.val == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.val == 1
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_maplist_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    mapped_list = MapList(func, lst)
    assert mapped_list.func == func
    assert mapped_list.list == lst


# LLM-generated content at query #27
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10
    assert r.val == 0

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10
    assert r.val == 1

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5
    assert r.val == 1

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_constructor_initialization():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #29
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_constructor_creates_maplist_with_given_function_and_list():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    mapped_list = MapList(func, lst)
    assert mapped_list.func == func
    assert mapped_list.list == lst


# LLM-generated content at query #31
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_lazy_list_constructor_initializes_correctly():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.exhausted is False
    assert lazy_list.list == []
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #33
#--------------------------

```python
def test_MapList_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #34
#--------------------------

```python
def test_constructor_with_valid_inputs():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst

def test_constructor_with_empty_list():
    func = lambda x: x * 2
    lst = []
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst

def test_constructor_with_none_func():
    func = None
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst

def test_constructor_with_none_list():
    func = lambda x: x * 2
    lst = None
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #35
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_lazy_list_constructor_initializes_correctly():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #38
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #39
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #40
#--------------------------

```python
def test_constructor_creates_maplist_with_given_function_and_list():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #41
#--------------------------

```python
def test_constructor_initialization():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    mapped_list = MapList(func, lst)
    assert mapped_list.func == func
    assert mapped_list.list == lst


# LLM-generated content at query #42
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_drop_until_empty_iterable():
    assert list(drop_until(lambda x: x > 5, [])) == []

def test_drop_until_all_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_no_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 10, [1, 2, 3])) == []

def test_drop_until_some_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]

def test_drop_until_first_element_satisfies_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_last_element_satisfies_predicate():
    assert list(drop_until(lambda x: x > 5, [1, 2, 3, 4, 5, 6])) == [6]

def test_drop_until_with_string_iterable():
    assert list(drop_until(lambda x: x == 'c', 'abcdef')) == ['c', 'd', 'e', 'f']

def test_drop_until_with_tuple_iterable():
    assert list(drop_until(lambda x: x > 3, (1, 2, 3, 4, 5))) == [4, 5]


# LLM-generated content at query #44
#--------------------------

```python
def test_lazy_list_constructor_initializes_correctly():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert lazy_list.iter is iter(iterable)


# LLM-generated content at query #45
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #46
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError for no args"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError for too many args"
    except ValueError:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.val == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.val == 1
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #48
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #49
#--------------------------

```python
def test_drop_until_with_empty_iterable():
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []

def test_drop_until_with_all_elements_satisfy_predicate():
    result = list(drop_until(lambda x: x > 0, [1, 2, 3, 4]))
    assert result == [1, 2, 3, 4]

def test_drop_until_with_no_elements_satisfy_predicate():
    result = list(drop_until(lambda x: x > 10, [1, 2, 3, 4]))
    assert result == []

def test_drop_until_with_some_elements_satisfy_predicate():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]

def test_drop_until_with_first_element_satisfies_predicate():
    result = list(drop_until(lambda x: x == 1, [1, 2, 3, 4]))
    assert result == [1, 2, 3, 4]

def test_drop_until_with_last_element_satisfies_predicate():
    result = list(drop_until(lambda x: x == 4, [1, 2, 3, 4]))
    assert result == [4]


# LLM-generated content at query #50
#--------------------------

```python
def test_drop_until_basic_case():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]

def test_drop_until_empty_iterable():
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []

def test_drop_until_all_elements_satisfy():
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]

def test_drop_until_no_elements_satisfy():
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []

def test_drop_until_first_element_satisfies():
    result = list(drop_until(lambda x: x == 0, range(5)))
    assert result == [0, 1, 2, 3, 4]

def test_drop_until_last_element_satisfies():
    result = list(drop_until(lambda x: x == 4, range(5)))
    assert result == [4]

def test_drop_until_with_strings():
    result = list(drop_until(lambda s: len(s) > 3, ["a", "bb", "ccc", "dddd", "eeee"]))
    assert result == ["dddd", "eeee"]


# LLM-generated content at query #51
#--------------------------

```python
def test_drop_until_predicate_true():
    predicate_called = False
    def pred_fn(x):
        nonlocal predicate_called
        predicate_called = True
        return True
    list(drop_until(pred_fn, [1, 2, 3]))
    assert predicate_called


# LLM-generated content at query #52
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #53
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #54
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #55
#--------------------------

```python
def test_drop_until_with_empty_iterable():
    assert list(drop_until(lambda x: x > 5, [])) == []

def test_drop_until_with_all_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_with_no_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 10, [1, 2, 3])) == []

def test_drop_until_with_some_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]

def test_drop_until_with_first_element_satisfying_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_with_last_element_satisfying_predicate():
    assert list(drop_until(lambda x: x > 2, [1, 2, 3])) == [3]


# LLM-generated content at query #56
#--------------------------

```python
def test_drop_until_predicate_false():
    result = list(drop_until(lambda x: x > 5, [1, 2, 3, 4, 5]))
    assert result == [1, 2, 3, 4, 5]


# LLM-generated content at query #57
#--------------------------

```python
def test_drop_until_predicate_false():
    predicate_called = False
    def pred_fn(x):
        nonlocal predicate_called
        predicate_called = True
        return False

    result = list(drop_until(pred_fn, [1, 2, 3]))
    assert predicate_called
    assert result == []


# LLM-generated content at query #58
#--------------------------

```python
def test_drop_until_predicate_evaluates_to_true():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]


# LLM-generated content at query #59
#--------------------------

```python
def test_drop_until_with_empty_iterable():
    assert list(drop_until(lambda x: x > 5, [])) == []

def test_drop_until_with_no_matching_element():
    assert list(drop_until(lambda x: x > 10, range(5))) == []

def test_drop_until_with_first_element_matching():
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]

def test_drop_until_with_middle_element_matching():
    assert list(drop_until(lambda x: x > 2, range(5))) == [3, 4]

def test_drop_until_with_last_element_matching():
    assert list(drop_until(lambda x: x > 3, range(5))) == [4]

def test_drop_until_with_all_elements_matching():
    assert list(drop_until(lambda x: x >= 0, range(5))) == [0, 1, 2, 3, 4]

def test_drop_until_with_string_iterable():
    assert list(drop_until(lambda x: x == 'c', ['a', 'b', 'c', 'd'])) == ['c', 'd']


# LLM-generated content at query #60
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    iterable = [1, 2, 3, 4, 5]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert isinstance(lazy_list.iter, type(iter(iterable)))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_take_first_n_elements():
    assert list(take(5, range(10))) == [0, 1, 2, 3, 4]

def test_take_zero_elements():
    assert list(take(0, range(10))) == []

def test_take_more_elements_than_available():
    assert list(take(10, range(5))) == [0, 1, 2, 3, 4]

def test_take_negative_n_raises_error():
    try:
        list(take(-1, range(10)))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_take_with_empty_iterable():
    assert list(take(5, [])) == []

def test_take_with_string_iterable():
    assert list(take(3, "hello")) == ['h', 'e', 'l']


# LLM-generated content at query #2
#--------------------------

```python
def test_drop_basic_case():
    result = list(drop(3, [1, 2, 3, 4, 5]))
    assert result == [4, 5]

def test_drop_empty_iterable():
    result = list(drop(2, []))
    assert result == []

def test_drop_n_larger_than_iterable():
    result = list(drop(10, [1, 2, 3]))
    assert result == []

def test_drop_zero_elements():
    result = list(drop(0, [1, 2, 3]))
    assert result == [1, 2, 3]

def test_drop_negative_n_raises_error():
    try:
        list(drop(-1, [1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_drop_with_generator():
    result = list(drop(2, (x for x in range(5))))
    assert result == [2, 3, 4]

def test_drop_with_string():
    result = list(drop(3, "hello"))
    assert result == ['l', 'o']

def test_drop_with_range():
    result = list(drop(5, range(10)))
    assert result == [5, 6, 7, 8, 9]


# LLM-generated content at query #3
#--------------------------

```python
def test_getitem_single_index():
    lazy_list = LazyList([1, 2, 3, 4, 5])
    assert lazy_list[0] == 1
    assert lazy_list[2] == 3
    assert lazy_list[4] == 5

def test_getitem_negative_index():
    lazy_list = LazyList([1, 2, 3, 4, 5])
    assert lazy_list[-1] == 5
    assert lazy_list[-3] == 3

def test_getitem_slice():
    lazy_list = LazyList([1, 2, 3, 4, 5])
    assert lazy_list[1:4] == [2, 3, 4]
    assert lazy_list[:3] == [1, 2, 3]
    assert lazy_list[2:] == [3, 4, 5]
    assert lazy_list[:] == [1, 2, 3, 4, 5]

def test_getitem_empty_slice():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list[2:2] == []
    assert lazy_list[5:10] == []

def test_getitem_exhausted_list():
    lazy_list = LazyList([1, 2, 3])
    _ = lazy_list[2]  # Exhaust the list
    assert lazy_list[0] == 1
    assert lazy_list[1] == 2
    assert lazy_list[2] == 3


# LLM-generated content at query #4
#--------------------------

```python
def test___next___basic_iteration():
    r = Range(1, 5)
    assert next(r) == 1
    assert next(r) == 2
    assert next(r) == 3
    assert next(r) == 4

def test___next___with_step():
    r = Range(0, 10, 2)
    assert next(r) == 0
    assert next(r) == 2
    assert next(r) == 4
    assert next(r) == 6
    assert next(r) == 8

def test___next___stop_iteration():
    r = Range(1, 3)
    assert next(r) == 1
    assert next(r) == 2
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass

def test___next___negative_start():
    r = Range(-3, 1)
    assert next(r) == -3
    assert next(r) == -2
    assert next(r) == -1
    assert next(r) == 0

def test___next___single_element():
    r = Range(5, 6)
    assert next(r) == 5
    try:
        next(r)
        assert False, "Expected StopIteration"
    except StopIteration:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_split_by_criterion():
    result = list(split_by(range(10), criterion=lambda x: x % 3 == 0))
    assert result == [[1, 2], [4, 5], [7, 8]]

def test_split_by_separator():
    result = list(split_by("Split by: ", separator=' '))
    assert result == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']]

def test_split_by_empty_segments():
    result = list(split_by(" Split by: ", empty_segments=True, separator=' '))
    assert result == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']]

def test_split_by_empty_segments_with_criterion():
    result = list(split_by([1, 2, 3, 4, 5], empty_segments=True, criterion=lambda x: x % 2 == 0))
    assert result == [[1], [], [3], [], [5]]

def test_split_by_empty_iterable():
    result = list(split_by([], separator=' '))
    assert result == [[]]

def test_split_by_empty_iterable_with_empty_segments():
    result = list(split_by([], empty_segments=True, separator=' '))
    assert result == [[]]

def test_split_by_no_criterion_or_separator():
    try:
        list(split_by([1, 2, 3]))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_split_by_both_criterion_and_separator():
    try:
        list(split_by([1, 2, 3], criterion=lambda x: x == 1, separator=2))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_chunk_with_empty_iterable():
    assert list(chunk(3, [])) == []

def test_chunk_with_n_equal_to_1():
    assert list(chunk(1, [1, 2, 3])) == [[1], [2], [3]]

def test_chunk_with_n_larger_than_iterable_length():
    assert list(chunk(5, [1, 2, 3])) == [[1, 2, 3]]

def test_chunk_with_n_smaller_than_iterable_length():
    assert list(chunk(2, [1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]

def test_chunk_with_exact_divisible_length():
    assert list(chunk(3, [1, 2, 3, 4, 5, 6])) == [[1, 2, 3], [4, 5, 6]]

def test_chunk_with_non_divisible_length():
    assert list(chunk(3, [1, 2, 3, 4, 5, 6, 7])) == [[1, 2, 3], [4, 5, 6], [7]]

def test_chunk_with_negative_n():
    try:
        list(chunk(-1, [1, 2, 3]))
    except ValueError as e:
        assert str(e) == "`n` should be positive"

def test_chunk_with_zero_n():
    try:
        list(chunk(0, [1, 2, 3]))
    except ValueError as e:
        assert str(e) == "`n` should be positive"


# LLM-generated content at query #7
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_drop_until_with_empty_iterable():
    assert list(drop_until(lambda x: x > 5, [])) == []

def test_drop_until_with_all_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_with_no_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 10, [1, 2, 3])) == []

def test_drop_until_with_some_elements_satisfying_predicate():
    assert list(drop_until(lambda x: x > 5, [1, 2, 3, 4, 5, 6, 7, 8, 9])) == [6, 7, 8, 9]

def test_drop_until_with_first_element_satisfying_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3])) == [1, 2, 3]

def test_drop_until_with_last_element_satisfying_predicate():
    assert list(drop_until(lambda x: x > 5, [1, 2, 3, 4, 5, 6])) == [6]

def test_drop_until_with_string_iterable():
    assert list(drop_until(lambda x: x == 'c', 'abcdef')) == ['c', 'd', 'e', 'f']

def test_drop_until_with_tuple_iterable():
    assert list(drop_until(lambda x: x > 3, (1, 2, 3, 4, 5))) == [4, 5]


# LLM-generated content at query #9
#--------------------------

```python
def test_getitem_single_index():
    r = Range(1, 11, 2)
    assert r[0] == 1
    assert r[2] == 5
    assert r[4] == 9

def test_getitem_negative_index():
    r = Range(1, 11, 2)
    assert r[-1] == 9
    assert r[-3] == 5

def test_getitem_slice():
    r = Range(1, 11, 2)
    assert r[1:4] == [3, 5, 7]
    assert r[:3] == [1, 3, 5]
    assert r[2:] == [5, 7, 9]
    assert r[::2] == [1, 5, 9]

def test_getitem_empty_slice():
    r = Range(1, 11, 2)
    assert r[5:2] == []
    assert r[10:20] == []

def test_getitem_step_in_slice():
    r = Range(1, 11, 1)
    assert r[1:8:2] == [2, 4, 6, 8]
    assert r[::3] == [1, 4, 7, 10]


# LLM-generated content at query #10
#--------------------------

```python
def test_drop_until_empty_iterable():
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []

def test_drop_until_all_elements_dropped():
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []

def test_drop_until_no_elements_dropped():
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]

def test_drop_until_some_elements_dropped():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]

def test_drop_until_with_string_predicate():
    result = list(drop_until(lambda s: len(s) > 3, ["a", "ab", "abc", "abcd", "abcde"]))
    assert result == ["abcd", "abcde"]

def test_drop_until_with_custom_objects():
    class Item:
        def __init__(self, value):
            self.value = value
    items = [Item(1), Item(2), Item(3), Item(4)]
    result = list(drop_until(lambda item: item.value > 2, items))
    assert len(result) == 2
    assert result[0].value == 3
    assert result[1].value == 4


# LLM-generated content at query #11
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert not lazy_list.exhausted


# LLM-generated content at query #12
#--------------------------

```python
def test_getitem_with_int_index():
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * x, lst)
    assert map_list[0] == 1
    assert map_list[1] == 4
    assert map_list[2] == 9
    assert map_list[3] == 16
    assert map_list[4] == 25

def test_getitem_with_negative_int_index():
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * x, lst)
    assert map_list[-1] == 25
    assert map_list[-2] == 16
    assert map_list[-3] == 9
    assert map_list[-4] == 4
    assert map_list[-5] == 1

def test_getitem_with_slice():
    lst = [1, 2, 3, 4, 5]
    map_list = MapList(lambda x: x * x, lst)
    assert map_list[1:4] == [4, 9, 16]
    assert map_list[:3] == [1, 4, 9]
    assert map_list[2:] == [9, 16, 25]
    assert map_list[::2] == [1, 9, 25]
    assert map_list[1:4:2] == [4, 16]


# LLM-generated content at query #13
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_maplist_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3, 4, 5]
    mapped_list = MapList(func, lst)
    assert mapped_list.func == func
    assert mapped_list.list == lst


# LLM-generated content at query #15
#--------------------------

```python
def test_constructor_creates_maplist_instance():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert isinstance(map_list, MapList)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #16
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, iter)

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, iter)


# LLM-generated content at query #19
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, Iterator)

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, Iterator)


# LLM-generated content at query #20
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')

def test_lazy_list_constructor_with_non_empty_iterable():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')

def test_lazy_list_constructor_with_generator():
    lazy_list = LazyList(x for x in range(5))
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #22
#--------------------------

```python
def test_maplist_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    mapped_list = MapList(func, lst)
    assert mapped_list.func == func
    assert mapped_list.list == lst


# LLM-generated content at query #23
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_constructor_creates_maplist_with_given_function_and_list():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #28
#--------------------------

```python
def test_drop_until_basic_case():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]

def test_drop_until_empty_iterable():
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []

def test_drop_until_all_elements_satisfy():
    result = list(drop_until(lambda x: x > 0, [1, 2, 3, 4]))
    assert result == [1, 2, 3, 4]

def test_drop_until_no_elements_satisfy():
    result = list(drop_until(lambda x: x > 10, [1, 2, 3, 4]))
    assert result == []

def test_drop_until_first_element_satisfies():
    result = list(drop_until(lambda x: x == 1, [1, 2, 3, 4]))
    assert result == [1, 2, 3, 4]

def test_drop_until_last_element_satisfies():
    result = list(drop_until(lambda x: x == 4, [1, 2, 3, 4]))
    assert result == [4]

def test_drop_until_with_strings():
    result = list(drop_until(lambda s: len(s) > 3, ["a", "ab", "abc", "abcd", "abcde"]))
    assert result == ["abcd", "abcde"]

def test_drop_until_with_custom_objects():
    class Item:
        def __init__(self, value):
            self.value = value
    items = [Item(1), Item(2), Item(3), Item(4)]
    result = list(drop_until(lambda item: item.value > 2, items))
    assert result == [items[2], items[3]]


# LLM-generated content at query #29
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_constructor_with_empty_list():
    func = lambda x: x * 2
    lst = []
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst

def test_constructor_with_non_empty_list():
    func = lambda x: x.upper()
    lst = ["hello", "world"]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #31
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #32
#--------------------------

```python
def test_drop_until_predicate_evaluates_to_true():
    predicate_called = False
    def mock_predicate(x):
        nonlocal predicate_called
        predicate_called = True
        return True

    result = list(drop_until(mock_predicate, [1, 2, 3]))
    assert predicate_called


# LLM-generated content at query #33
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.val == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.val == 1
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #35
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #36
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_maplist_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    maplist = MapList(func, lst)
    assert maplist.func == func
    assert maplist.list == lst


# LLM-generated content at query #38
#--------------------------

```python
def test_drop_until_predicate_true():
    predicate_called = False
    def pred_fn(x):
        nonlocal predicate_called
        predicate_called = True
        return x > 5

    result = list(drop_until(pred_fn, [6, 7, 8]))
    assert predicate_called is True
    assert result == [6, 7, 8]


# LLM-generated content at query #39
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    lazy_list = LazyList([1, 2, 3])
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #40
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_lazy_list_constructor_initialization():
    iterable = [1, 2, 3, 4, 5]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert lazy_list.exhausted is False
    assert hasattr(lazy_list, 'iter')


# LLM-generated content at query #42
#--------------------------

```python
def test_lazy_list_constructor_with_empty_iterable():
    lazy_list = LazyList([])
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, Iterator)

def test_lazy_list_constructor_with_non_empty_iterable():
    iterable = [1, 2, 3]
    lazy_list = LazyList(iterable)
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, Iterator)

def test_lazy_list_constructor_with_generator():
    def gen():
        yield 1
        yield 2
    lazy_list = LazyList(gen())
    assert lazy_list.list == []
    assert not lazy_list.exhausted
    assert isinstance(lazy_list.iter, Iterator)


# LLM-generated content at query #43
#--------------------------

```python
def test_drop_until_empty_iterable():
    assert list(drop_until(lambda x: x > 5, [])) == []

def test_drop_until_all_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3, 4])) == [1, 2, 3, 4]

def test_drop_until_no_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 10, [1, 2, 3, 4])) == []

def test_drop_until_some_elements_satisfy_predicate():
    assert list(drop_until(lambda x: x > 5, range(10))) == [6, 7, 8, 9]

def test_drop_until_first_element_satisfies_predicate():
    assert list(drop_until(lambda x: x > 0, [1, 2, 3, 4])) == [1, 2, 3, 4]

def test_drop_until_last_element_satisfies_predicate():
    assert list(drop_until(lambda x: x > 3, [1, 2, 3, 4])) == [4]

def test_drop_until_with_strings():
    assert list(drop_until(lambda s: len(s) > 3, ["a", "bb", "ccc", "dddd", "eeee"])) == ["dddd", "eeee"]


# LLM-generated content at query #44
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_drop_until_basic_case():
    result = list(drop_until(lambda x: x > 5, range(10)))
    assert result == [6, 7, 8, 9]

def test_drop_until_empty_iterable():
    result = list(drop_until(lambda x: x > 5, []))
    assert result == []

def test_drop_until_all_elements_satisfy():
    result = list(drop_until(lambda x: x >= 0, range(5)))
    assert result == [0, 1, 2, 3, 4]

def test_drop_until_no_elements_satisfy():
    result = list(drop_until(lambda x: x > 10, range(5)))
    assert result == []

def test_drop_until_first_element_satisfies():
    result = list(drop_until(lambda x: x == 0, range(5)))
    assert result == [0, 1, 2, 3, 4]

def test_drop_until_with_strings():
    result = list(drop_until(lambda s: len(s) > 3, ["a", "ab", "abc", "abcd", "abcde"]))
    assert result == ["abcd", "abcde"]

def test_drop_until_with_custom_objects():
    class Item:
        def __init__(self, value):
            self.value = value
    items = [Item(1), Item(2), Item(3), Item(4)]
    result = list(drop_until(lambda item: item.value > 2, items))
    assert len(result) == 2
    assert result[0].value == 3
    assert result[1].value == 4


# LLM-generated content at query #46
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 10, 2)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 2
    assert r.length == 4

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_range_constructor_too_many_args():
    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #48
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.val == 0
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 11)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 1
    assert r.val == 1
    assert r.length == 10

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.val == 1
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #49
#--------------------------

```python
def test_constructor():
    func = lambda x: x * 2
    lst = [1, 2, 3]
    map_list = MapList(func, lst)
    assert map_list.func == func
    assert map_list.list == lst


# LLM-generated content at query #50
#--------------------------

```python
def test_range_constructor_single_arg():
    r = Range(10)
    assert r.l == 0
    assert r.r == 10
    assert r.step == 1
    assert r.length == 10

def test_range_constructor_two_args():
    r = Range(1, 10)
    assert r.l == 1
    assert r.r == 10
    assert r.step == 1
    assert r.length == 9

def test_range_constructor_three_args():
    r = Range(1, 11, 2)
    assert r.l == 1
    assert r.r == 11
    assert r.step == 2
    assert r.length == 5

def test_range_constructor_invalid_args():
    try:
        Range()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        Range(1, 2, 3, 4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


